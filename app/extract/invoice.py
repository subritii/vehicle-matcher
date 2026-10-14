"""Invoice file -> InvoiceFields, with rules only (no LLM).

Pipeline
--------
1. Text with positions. A PDF with a text layer is read with pdfplumber (words
   merged into phrases). Scans, photos and image-only PDFs go through OCR
   (PaddleOCR, Tesseract as fallback). Either way we get ``Seg`` boxes, which are
   grouped into ``Line`` rows after undoing the page's skew.
2. Fields. Each field has label patterns (e.g. "Odometer", "Lic Plate") and a
   parser for the value. A value is looked for after the label on the same line,
   then in the box directly under the label (form-style layouts, where a header row
   holds labels and the row below holds values). Labels have priorities, e.g.
   "Total Due" beats "Total" and "Mileage In" beats "Mileage".
3. VIN comes from app.extract.vin, which handles check digits, OCR repairs and split
   or glued VINs. Year and make fall back to what the VIN encodes.
4. PDF gaps. If a field is missing from a PDF's text layer (e.g. handwritten mileage),
   the page is rendered and OCR'd, and only the missing fields are filled from it.
5. Review and VIN trust (``_finalize``). The invoice goes to review when it has more
   than one trusted VIN, a VIN that is unlabeled/assembled, fails the check digit, was
   repaired or is ambiguous, an unknown manufacturer code (WMI), a year or make that
   contradicts the VIN, no VIN and no plate, or a VIN that isn't confirmed.
   A VIN is *trusted* (may auto-link) only with no such issue and at least one agreeing
   ``VinConfirmation``: "text_layer" for VINs from a digital text layer, "vpic" for OCR
   VINs whose vPIC decode has no problem codes and agrees with the year and make
   printed on the invoice.
   More sources (photo VIN) can be added with ``add_vin_confirmation``.
6. Optional vision model (``fill_gaps_with_vision``, app.extract.vision). Only fields
   the rules left blank or with low confidence are taken from it, after the same
   parsers the rules use. Its VIN must pass the check digit and either match an OCR
   reading or be confirmed by vPIC against the year and make the rules read; a VIN
   only the model read is never trusted. A model VIN that agrees with an OCR VIN adds
   a "vision" confirmation; a different valid VIN blocks trust.

Every field carries a confidence in (0, 1]: label strength times OCR score. Its
``source`` says where it came from: "pdf_text", "ocr:<engine>", "vin" (decoded from
the VIN) or "vision:<model>".
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Callable, Protocol

from app.extract import vin as vinlib

FIELD_NAMES = (
    "vin", "plate", "plate_state", "year", "make", "model", "color", "mileage",
    "invoice_date", "vendor", "invoice_no", "total",
)

# --- Positioned text --------------------------------------------------------------


@dataclass(frozen=True)
class Seg:
    """A box of text: a phrase from a PDF, or a text line from OCR."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    angle: float = 0.0  # baseline angle in degrees (OCR); 0 for PDFs

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class Line:
    """One row of text: segments left to right, joined with two spaces."""

    def __init__(self, segs: list[Seg]):
        self.segs = segs
        self.spans: list[tuple[int, int]] = []
        pos = 0
        for s in segs:
            self.spans.append((pos, pos + len(s.text)))
            pos += len(s.text) + 2
        self.text = "  ".join(s.text for s in segs)
        self.upper = self.text.upper()
        self.y = statistics.fmean((s.y0 + s.y1) / 2 for s in segs)

    def seg_index(self, pos: int) -> int:
        for i, (a, b) in enumerate(self.spans):
            if pos < b or (i + 1 < len(self.spans) and pos < self.spans[i + 1][0]):
                return i
        return len(self.spans) - 1

    def x_at(self, pos: int) -> float:
        i = self.seg_index(pos)
        a, b = self.spans[i]
        s = self.segs[i]
        frac = min(max((pos - a) / max(b - a, 1), 0.0), 1.0)
        return s.x0 + frac * (s.x1 - s.x0)

    def __repr__(self) -> str:
        return f"Line({self.text!r})"


def _center(s: Seg) -> tuple[float, float]:
    return (s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2


def _group_rows(segs: list[Seg]) -> list[list[Seg]]:
    """Group segments into rows, following the text's slope.

    A photo taken at an angle has tilted rows, and with perspective (keystone) the
    tilt changes down the page. So each segment uses the *local* baseline angle (the
    median angle of the 3 nearest wide segments) and joins the row whose
    nearest segment, projected along that angle, lands within half a line height.
    """
    if not segs:
        return []
    h = statistics.median(max(s.height, 1e-6) for s in segs)
    wide = [s for s in segs if (s.x1 - s.x0) > 3 * s.height]
    global_angle = statistics.median(w.angle for w in wide) if wide else 0.0

    def local_tan(s: Seg) -> float:
        cy = _center(s)[1]
        near = sorted(wide, key=lambda w: abs(_center(w)[1] - cy))[:3]
        near = [w.angle for w in near if abs(_center(w)[1] - cy) <= 6 * h]
        return math.tan(math.radians(statistics.median(near) if near else global_angle))

    slope = {id(s): local_tan(s) for s in segs}
    rows: list[list[Seg]] = []
    for s in sorted(segs, key=lambda s: _center(s)[1]):
        sx, sy = _center(s)
        best = None
        for ri in range(len(rows) - 1, max(len(rows) - 8, -1), -1):
            ref = min(rows[ri], key=lambda r: abs(_center(r)[0] - sx))
            rx, ry = _center(ref)
            d = abs(sy - (ry + slope[id(s)] * (sx - rx)))
            if d <= 0.5 * h and (best is None or d < best[0]):
                best = (d, ri)
        if best:
            rows[best[1]].append(s)
        else:
            rows.append([s])

    def row_key(row: list[Seg]) -> float:  # y where the row's slope crosses x = 0
        cx, cy = _center(row[0])
        return cy - slope[id(row[0])] * cx

    rows = [sorted(r, key=lambda s: s.x0) for r in rows]
    return sorted(rows, key=row_key)


def _merge_words(row: list[Seg]) -> list[Seg]:
    """Join words separated by about a space into phrases; wider gaps split columns."""
    out: list[Seg] = []
    for w in row:
        prev = out[-1] if out else None
        if prev and w.x0 - prev.x1 <= 0.8 * max(prev.height, w.height):
            out[-1] = Seg(f"{prev.text} {w.text}", prev.x0, min(prev.y0, w.y0), w.x1,
                          max(prev.y1, w.y1), min(prev.score, w.score), prev.angle)
        else:
            out.append(w)
    return out


@dataclass
class Page:
    lines: list[Line]
    width: float
    height: float
    source: str  # pdf_text | ocr:<engine> | text
    segs: list[Seg] = field(default_factory=list, repr=False)

    @classmethod
    def from_segs(cls, segs: list[Seg], width: float, height: float, source: str,
                  merge_words: bool = False) -> Page:
        rows = _group_rows([s for s in segs if s.text.strip()])
        if merge_words:
            rows = [_merge_words(r) for r in rows]
        return cls([Line(r) for r in rows], width, height, source, list(segs))

    @classmethod
    def from_text(cls, text: str, source: str = "text") -> Page:
        """Plain text -> page. Runs of 2+ spaces split segments; each character is 5
        units wide and each line 12 units tall, so columns line up like on paper."""
        segs = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for m in re.finditer(r"\S+(?: \S+)*", line):
                segs.append(Seg(m.group(), m.start() * 5, i * 12, m.end() * 5, i * 12 + 10))
        return cls.from_segs(segs, width=max((len(l) for l in lines), default=1) * 5,
                             height=max(len(lines), 1) * 12, source=source)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @functools.cached_property
    def median_height(self) -> float:
        return statistics.median(s.height for line in self.lines for s in line.segs) if self.lines else 10.0


# --- Reading files ----------------------------------------------------------------

MIN_TEXT_LAYER_CHARS = 25
OCR_DPI = 200
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


class OcrEngine(Protocol):
    name: str

    def __call__(self, image) -> list[Seg]: ...


class PaddleEngine:
    name = "paddle"
    word_level = False

    def __init__(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def __call__(self, image) -> list[Seg]:
        import numpy as np

        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        segs = []
        for r in self._ocr.predict(bgr):
            for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"]):
                (x0, y0), (x1, y1), _, (x3, y3) = [(float(p[0]), float(p[1])) for p in poly[:4]]
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                h = math.hypot(x3 - x0, y3 - y0)
                cy = statistics.fmean(ys)
                segs.append(Seg(text, min(xs), cy - h / 2, max(xs), cy + h / 2, float(score),
                                math.degrees(math.atan2(y1 - y0, x1 - x0))))
        return segs


class TesseractEngine:
    name = "tesseract"
    word_level = True

    def __init__(self) -> None:
        import pytesseract

        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract binary not found (brew install tesseract)")
        self._pt = pytesseract

    def __call__(self, image) -> list[Seg]:
        d = self._pt.image_to_data(image, output_type=self._pt.Output.DICT)
        segs = []
        for i, text in enumerate(d["text"]):
            conf = float(d["conf"][i])
            if text.strip() and conf >= 0:
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                segs.append(Seg(text, x, y, x + w, y + h, conf / 100))
        return segs


# --- OCR with a per-image timeout ---------------------------------------------------


class OcrFailed(RuntimeError):
    """OCR gave no result for an image; the invoice is treated as unreadable."""


class OcrTimeout(OcrFailed):
    pass


def ocr_timeout() -> float:
    """OCR_TIMEOUT: seconds allowed per image (default 120); 0 disables the worker."""
    raw = os.getenv("OCR_TIMEOUT", "").strip()
    return float(raw) if raw else 120.0


def _ocr_worker(engine_cls, conn) -> None:
    try:
        engine = engine_cls()
    except Exception as e:  # ImportError, missing binary, model download failure
        conn.send(("error", f"{getattr(engine_cls, 'name', engine_cls.__name__)}: {e}"))
        return
    conn.send(("ready", None))
    from PIL import Image

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        mode, size, data = msg
        try:
            conn.send(("ok", [asdict(s) for s in engine(Image.frombytes(mode, size, data))]))
        except Exception as e:
            conn.send(("error", repr(e)))


class TimedOcr:
    """Runs an OCR engine in one worker process and kills it when an image takes
    longer than ``timeout`` seconds (native OCR code can't be interrupted in-process).
    The next image starts a fresh worker. Model loading isn't timed."""

    def __init__(self, engine_cls: type, timeout: float) -> None:
        self.engine_cls = engine_cls
        self.timeout = timeout
        self.name = engine_cls.name
        self.word_level = bool(getattr(engine_cls, "word_level", False))
        self._proc = None
        self._conn = None

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_ocr_worker, args=(self.engine_cls, child), daemon=True)
        self._proc.start()
        child.close()
        while not self._conn.poll(1):
            if not self._proc.is_alive():
                self._kill()
                raise RuntimeError(f"{self.name}: OCR worker exited while loading")
        status, payload = self._conn.recv()
        if status != "ready":
            self._kill()
            raise RuntimeError(payload)

    def __call__(self, image) -> list[Seg]:
        self.start()
        image = image.convert("RGB")
        self._conn.send((image.mode, image.size, image.tobytes()))
        if not self._conn.poll(self.timeout):
            self._kill()
            raise OcrTimeout(f"no result after {self.timeout:g} s")
        try:
            status, payload = self._conn.recv()
        except EOFError:
            self._kill()
            raise OcrFailed("OCR worker crashed") from None
        if status != "ok":
            raise OcrFailed(payload)
        return [Seg(**s) for s in payload]

    def _kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join()
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            try:
                self._conn.send(None)
                self._proc.join(5)
            except (BrokenPipeError, OSError):
                pass
        self._kill()


@functools.cache
def get_ocr_engine() -> OcrEngine:
    """OCR_ENGINE=paddle (default) or tesseract; falls back to the other if the
    preferred one can't load. With OCR_TIMEOUT > 0 (default) the engine runs in a
    worker process with that per-image timeout."""
    preferred = os.getenv("OCR_ENGINE", "paddle").strip().lower() or "paddle"
    order = [PaddleEngine, TesseractEngine] if preferred == "paddle" else [TesseractEngine, PaddleEngine]
    timeout = ocr_timeout()
    errors = []
    for engine in order:
        try:
            if timeout > 0:
                timed = TimedOcr(engine, timeout)
                timed.start()
                return timed
            return engine()
        except Exception as e:  # ImportError, missing binary, model download failure
            errors.append(f"{engine.name}: {e}")
    raise RuntimeError("no OCR engine available: " + "; ".join(errors))


def _load_image(path: Path):
    from PIL import Image, ImageOps

    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _rasterize_pdf(path: Path) -> list:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        return [page.render(scale=OCR_DPI / 72).to_pil().convert("RGB") for page in doc]
    finally:
        doc.close()


def _pdf_text_page(path: Path) -> Page | None:
    import pdfplumber

    words: list[Seg] = []
    with pdfplumber.open(path) as pdf:
        y_off = 0.0
        width = pdf.pages[0].width if pdf.pages else 1
        for pg in pdf.pages:
            for w in pg.extract_words():
                words.append(Seg(w["text"], w["x0"], w["top"] + y_off, w["x1"], w["bottom"] + y_off))
            y_off += pg.height
    if sum(c.isalnum() for w in words for c in w.text) < MIN_TEXT_LAYER_CHARS:
        return None
    return Page.from_segs(words, width, y_off, "pdf_text", merge_words=True)


def _ocr_page(path: Path, ocr: OcrEngine, cache_dir: Path | None) -> Page:
    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        cache_file = Path(cache_dir) / f"{digest}-{ocr.name}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return Page.from_segs([Seg(**s) for s in data["segs"]], data["width"], data["height"],
                                  f"ocr:{ocr.name}", merge_words=data["word_level"])

    images = _rasterize_pdf(path) if path.suffix.lower() == ".pdf" else [_load_image(path)]
    segs: list[Seg] = []
    y_off = 0.0
    for im in images:
        segs += [Seg(s.text, s.x0, s.y0 + y_off, s.x1, s.y1 + y_off, s.score, s.angle) for s in ocr(im)]
        y_off += im.height
    width = max(im.width for im in images)
    word_level = bool(getattr(ocr, "word_level", False))
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"width": width, "height": y_off, "word_level": word_level,
                                          "segs": [asdict(s) for s in segs]}))
    return Page.from_segs(segs, width, y_off, f"ocr:{ocr.name}", merge_words=word_level)


def read_page(path: str | Path, ocr: OcrEngine | None = None, cache_dir: str | Path | None = None,
              force_ocr: bool = False) -> Page:
    """PDF text layer if there is one, otherwise OCR (``force_ocr`` OCRs a PDF even
    with a text layer). ``cache_dir`` stores OCR output keyed by file content."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf" and not force_ocr:
        page = _pdf_text_page(path)
        if page is not None:
            return page
    elif suffix not in IMAGE_SUFFIXES | {".pdf"}:
        raise ValueError(f"unsupported file type: {path.name}")
    return _ocr_page(path, ocr or get_ocr_engine(), Path(cache_dir) if cache_dir else None)


# --- Label/value search -----------------------------------------------------------


@dataclass
class Hit:
    value: object
    priority: int  # lower is stronger
    line: int
    confidence: float
    evidence: str


def find_labeled(page: Page, label_re: re.Pattern, parse: Callable[[str, str], object | None],
                 *, priority: int = 0, base_conf: float = 0.9, below: int = 2) -> list[Hit]:
    """Values for a label: after the label on the same line (rest of its segment plus
    the next segment), else in a segment under the label in the next ``below`` rows.
    ``parse(upper_text, original_text)`` returns a value or None."""
    hits = []
    tol = page.median_height
    for li, line in enumerate(page.lines):
        for m in label_re.finditer(line.upper):
            label_seg = line.seg_index(max(m.end() - 1, m.start()))
            stop = line.spans[min(label_seg + 1, len(line.spans) - 1)][1]
            rest = slice(m.end(), max(stop, m.end()))
            value = parse(line.upper[rest], line.text[rest])
            score = line.segs[label_seg].score
            if value is None and below:
                lx0, lx1 = line.x_at(m.start()), line.x_at(m.end())
                for nl in page.lines[li + 1: li + 1 + below]:
                    if nl.y - line.y > 4 * tol:
                        break
                    under = [s for s in nl.segs if s.x0 < lx1 + 0.3 * tol and s.x1 > lx0 - 0.3 * tol]
                    for s in under:
                        value = parse(s.text.upper(), s.text)
                        if value is not None:
                            score = min(score, s.score)
                            break
                    if value is not None:
                        break
            if value is not None:
                hits.append(Hit(value, priority, li, base_conf * score, line.text))
    return hits


def _labels(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


def _first(hits: list[Hit]) -> Hit | None:
    return min(hits, key=lambda h: (h.priority, h.line), default=None)


# --- Value parsers ----------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(
    ("JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
     "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"), start=1)}

_DATE_PATTERNS = (
    ("ymd", re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")),
    ("num", re.compile(r"(?<![\d/.-])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})(?![\d/])")),
    ("dmy", re.compile(r"(?<![\dA-Z])(\d{1,2})[\s-]+([A-Z]{3,9})\.?[\s,-]+(\d{4})(?!\d)", re.I)),
    ("mdy", re.compile(r"(?<![A-Z])([A-Z]{3,9})\.?\s+(\d{1,2})(?:ST|ND|RD|TH)?,?\s+(\d{4})(?!\d)", re.I)),
)


def _month(name: str) -> int | None:
    name = name.upper()
    for full, n in MONTHS.items():
        if len(name) >= 3 and full.startswith(name):
            return n
    return None


def _make_date(y: int, m: int | None, d: int) -> date | None:
    if m is None or not 1990 <= y <= 2099:
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def date_spans(text: str) -> list[tuple[int, int, date]]:
    out = []
    for kind, rx in _DATE_PATTERNS:
        for m in rx.finditer(text):
            a, b, c = m.groups()
            if kind == "ymd":
                dt = _make_date(int(a), int(b), int(c))
            elif kind == "num":
                y = int(c) + (2000 if len(c) == 2 else 0)
                first, second = int(a), int(b)
                dt = _make_date(y, first, second) if first <= 12 else _make_date(y, second, first)
            elif kind == "dmy":
                dt = _make_date(int(c), _month(b), int(a))
            else:
                dt = _make_date(int(c), _month(a), int(b))
            if dt:
                out.append((m.start(), m.end(), dt))
    return sorted(out)


def parse_date(text: str) -> date | None:
    spans = date_spans(text)
    return spans[0][2] if spans else None


MONEY_RE = re.compile(r"(?<![\d.,])-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)\.(\d{2})(?![\d])")


def _money_values(text: str) -> list[Decimal]:
    return [Decimal(m.group(1).replace(",", "") + "." + m.group(2)) for m in MONEY_RE.finditer(text)]


def _masked(text: str) -> str:
    """Blank out dates and money so their digits aren't read as other numbers."""
    chars = list(text)
    spans = [(a, b) for a, b, _ in date_spans(text)] + [m.span() for m in MONEY_RE.finditer(text)]
    for a, b in spans:
        chars[a:b] = " " * (b - a)
    return "".join(chars)


_MILEAGE_VALUE_RE = re.compile(
    r"""^[\s:#=.\-]*                                 # separators after the label
        (?:/\s*OUT[\s:#=.\-]*)?                       # "ODO IN/OUT 33100/33104"
        (?:\((?:MI|MILES)\)[\s:#=.\-]*)?              # "Odometer (mi): ..."
        (\d{1,3}(?:[,.]\d{3})+|\d{1,7})                # the number; "." may be an OCR'd ","
        (?:\s*(?:MI|MILES|MLS)\.?)?                     # optional unit
        (?:/\d{1,3}(?:[,.]\d{3})+|/\d{1,7})?            # the OUT reading of "IN/OUT"
        (?=$|[;|)]|,(?!\d)|\s+(?!\d))                  # then a clean break, not another number
    """,
    re.X,
)


def parse_mileage(text: str) -> int | None:
    """Mileage printed right after its label (or alone in the box under it).

    Only the first thing after the label counts: if that isn't a clean number (blank,
    another field's label, digits glued to letters, an OCR misread, a date or money)
    the result is None, so a neighboring field's digits are never borrowed."""
    m = _MILEAGE_VALUE_RE.match(text)
    if not m:
        return None
    n = int(re.sub(r"[,.]", "", m.group(1)))
    return n if 1 <= n <= 999_999 else None


COLOR_WORDS = {
    "white": ("WHITE", "WHT", "WHI"),
    "black": ("BLACK", "BLK"),
    "silver": ("SILVER", "SIL", "SLV", "SILV"),
    "gray": ("GRAY", "GREY", "GRY", "GUNMETAL", "CHARCOAL", "GRAPHITE"),
    "blue": ("BLUE", "BLU", "NAVY"),
    "red": ("RED", "MAROON", "BURGUNDY", "CRIMSON"),
    "green": ("GREEN", "GRN"),
    "brown": ("BROWN", "BRN"),
    "beige": ("BEIGE", "TAN"),
    "gold": ("GOLD",),
    "yellow": ("YELLOW", "YEL"),
    "orange": ("ORANGE", "ORG"),
    "purple": ("PURPLE",),
}
_COLOR_LOOKUP = {w: c for c, words in COLOR_WORDS.items() for w in words}


def normalize_color(raw: str) -> str | None:
    """Printed color -> basic color name; None if no known color word."""
    for tok in re.findall(r"[A-Z]+", raw.upper())[:4]:
        if tok in _COLOR_LOOKUP:
            return _COLOR_LOOKUP[tok]
    return None


US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH "
    "NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split()
)


_STATE_OCR = {"1": "IL", "0": "OD", "5": "S", "8": "B", "2": "Z", "6": "G"}


def _ocr_state(tok: str) -> str | None:
    """'11' -> IL, '0H' -> OH: a 2-char token with OCR digits that maps to exactly one state."""
    if len(tok) != 2 or tok.isalpha():
        return None
    options = {""}
    for c in tok:
        options = {o + r for o in options for r in _STATE_OCR.get(c, c)}
    states = options & US_STATES
    return states.pop() if len(states) == 1 else None


def parse_plate(text: str) -> tuple[str, str | None, bool] | None:
    """'7KJD221 CA', 'TX GHT4412', 'ABC1234/NY', 'XVB9960 (WA)', 'ABC 1234 FL'.
    Returns (plate, state, state_guessed_from_ocr_digits)."""
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[A-Z0-9]+", text.upper())][:5]
    for i, (tok, start, end) in enumerate(toks):
        if not (any(c.isdigit() for c in tok) and 2 <= len(tok) <= 8):
            continue
        first = i
        plate = tok
        if i > 0:  # "ABC 1234": letters joined to the digits
            prev, ps, pe = toks[i - 1]
            gap = text[pe:start]
            if prev.isalpha() and prev not in US_STATES and len(prev) <= 4 and len(prev + tok) <= 8 \
                    and re.fullmatch(r"[ -]?", gap):
                plate, first = prev + tok, i - 1
        before = toks[first - 1][0] if first > 0 else ""
        after = toks[i + 1][0] if i + 1 < len(toks) else ""
        for tok in (before, after):
            if tok in US_STATES:
                return plate, tok, False
        guess = _ocr_state(after) or _ocr_state(before)
        return plate, guess, guess is not None
    return None


# --- Vehicle (year/make/model) ----------------------------------------------------

MAKES = {
    "ACURA": "Acura", "AUDI": "Audi", "BMW": "BMW", "BUICK": "Buick", "CADILLAC": "Cadillac",
    "CHEVROLET": "Chevrolet", "CHEVY": "Chevrolet", "CHEV": "Chevrolet", "CHRYSLER": "Chrysler",
    "DODGE": "Dodge", "FIAT": "Fiat", "FORD": "Ford", "GENESIS": "Genesis", "GMC": "GMC",
    "HONDA": "Honda", "HYUNDAI": "Hyundai", "INFINITI": "Infiniti", "JAGUAR": "Jaguar", "JEEP": "Jeep",
    "KIA": "Kia", "LAND ROVER": "Land Rover", "LEXUS": "Lexus", "LINCOLN": "Lincoln", "MAZDA": "Mazda",
    "MERCEDES-BENZ": "Mercedes-Benz", "MERCEDES BENZ": "Mercedes-Benz", "MERCEDES": "Mercedes-Benz",
    "MINI": "Mini", "MITSUBISHI": "Mitsubishi", "NISSAN": "Nissan", "PORSCHE": "Porsche", "RAM": "Ram",
    "SUBARU": "Subaru", "TESLA": "Tesla", "TOYOTA": "Toyota", "VOLKSWAGEN": "Volkswagen", "VW": "Volkswagen",
    "VOLVO": "Volvo",
}
_MAKE_RE = re.compile(
    r"(?<![A-Z0-9])(" + "|".join(re.escape(k) for k in sorted(MAKES, key=len, reverse=True)) + r")(?![A-Z0-9])"
)

# Common US-market models (general knowledge, not exhaustive). Unknown models still
# extract, at lower confidence.
MODELS = {
    "Acura": "MDX RDX TLX ILX Integra",
    "Audi": "A3|A4|A5|A6|Q3|Q5|Q7|Q8|e-tron",
    "BMW": "328i|330i|530i|540i|X1|X3|X5|X7|M3|M5|3 Series|5 Series|i4",
    "Buick": "Enclave|Encore|Encore GX|Envision|LaCrosse",
    "Cadillac": "Escalade|XT4|XT5|XT6|CT4|CT5",
    "Chevrolet": "Silverado 1500|Silverado 2500HD|Silverado 3500HD|Silverado|Equinox|Malibu|Tahoe|Suburban|"
                 "Traverse|Colorado|Camaro|Corvette|Cruze|Impala|Blazer|Trailblazer|Trax|Bolt EV|Spark|Express",
    "Chrysler": "Pacifica|300|Town & Country|Voyager",
    "Dodge": "Charger|Challenger|Durango|Grand Caravan|Journey|Hornet",
    "Fiat": "500|500X",
    "Ford": "F-150|F-250|F-350|Escape|Explorer|Edge|Expedition|Mustang|Mustang Mach-E|Fusion|Focus|Ranger|"
            "Bronco|Bronco Sport|Maverick|Transit|Transit Connect|EcoSport|Taurus|Flex",
    "Genesis": "G70|G80|G90|GV70|GV80",
    "GMC": "Sierra 1500|Sierra 2500HD|Sierra|Yukon|Yukon XL|Acadia|Terrain|Canyon|Savana",
    "Honda": "Accord|Civic|CR-V|HR-V|Pilot|Odyssey|Ridgeline|Passport|Fit|Insight",
    "Hyundai": "Elantra|Sonata|Tucson|Santa Fe|Kona|Palisade|Accent|Venue|Ioniq 5|Santa Cruz",
    "Infiniti": "Q50|Q60|QX50|QX60|QX80",
    "Jaguar": "F-Pace|E-Pace|XF",
    "Jeep": "Grand Cherokee|Cherokee|Wrangler|Compass|Renegade|Gladiator|Wagoneer|Grand Wagoneer|Liberty|Patriot",
    "Kia": "Optima|K5|Forte|Sorento|Sportage|Soul|Telluride|Seltos|Rio|Carnival|Sedona|Niro|EV6",
    "Land Rover": "Range Rover|Range Rover Sport|Discovery|Defender",
    "Lexus": "RX 350|RX350|ES 350|ES350|NX 300|IS 300|GX 460|RX|ES|NX|IS|GX",
    "Lincoln": "Navigator|Aviator|Corsair|Nautilus|MKZ",
    "Mazda": "CX-5|CX-30|CX-9|CX-50|CX-90|Mazda3|Mazda6|MX-5 Miata|MX-5|3|6",
    "Mercedes-Benz": "C-Class|E-Class|S-Class|GLC|GLE|GLA|GLB|GLS|C300|E350|Sprinter",
    "Mini": "Cooper|Countryman",
    "Mitsubishi": "Outlander|Outlander Sport|Mirage|Eclipse Cross",
    "Nissan": "Altima|Sentra|Rogue|Rogue Sport|Pathfinder|Murano|Frontier|Titan|Maxima|Versa|Kicks|Armada|Leaf",
    "Porsche": "Cayenne|Macan|911|Panamera|Taycan",
    "Ram": "1500|2500|3500|ProMaster|ProMaster City",
    "Subaru": "Outback|Forester|Crosstrek|Impreza|Legacy|Ascent|WRX|BRZ",
    "Tesla": "Model 3|Model Y|Model S|Model X|Cybertruck",
    "Toyota": "Camry|Corolla|Corolla Cross|RAV4|Highlander|Tacoma|Tundra|Prius|4Runner|Sienna|Avalon|Sequoia|"
              "Venza|C-HR|GR86",
    "Volkswagen": "Jetta|Passat|Golf|GTI|Tiguan|Atlas|Atlas Cross Sport|Taos|ID.4|Beetle",
    "Volvo": "XC40|XC60|XC90|S60|S90|V60",
}
MODELS = {
    make: {re.sub(r"[^A-Z0-9]", "", m.upper()): m for m in (v.split("|") if "|" in v else v.split())}
    for make, v in MODELS.items()
}
# OCR swaps look-alike letters and digits ("330i" -> "3301", "CX-5" -> "CX-S"), so models
# also match after folding both sides to digits.
_OCR_FOLD = str.maketrans("IOLSBZG", "1015826")
MODELS_FOLDED = {make: {k.translate(_OCR_FOLD): m for k, m in models.items()} for make, models in MODELS.items()}

YEAR_RE = re.compile(r"(?<![\d/.\-,$#])(19[89]\d|20[0-3]\d)(?![\d/.\-,])")
_TOKEN_RE = re.compile(r"[A-Z0-9][A-Z0-9.\-&]*")


def match_model(upper: str, original: str, make: str) -> tuple[str, str] | None:
    """Model at the start of ``upper`` -> (name, kind). kind is "known" (longest match
    over up to 4 tokens), "ocr" (matches after folding OCR look-alikes) or "unknown"
    (just the next word)."""
    toks = list(_TOKEN_RE.finditer(upper))[:4]
    known, folded = MODELS.get(make, {}), MODELS_FOLDED.get(make, {})
    for k in range(len(toks), 0, -1):
        key = re.sub(r"[^A-Z0-9]", "", "".join(t.group() for t in toks[:k]))
        if key in known:
            return known[key], "known"
        if key.translate(_OCR_FOLD) in folded:
            return folded[key.translate(_OCR_FOLD)], "ocr"
    if toks:
        t = toks[0]
        word = original[t.start():t.end()].strip(".,")
        if len(word) >= 2 and not YEAR_RE.fullmatch(word) and any(c.isalpha() for c in word):
            return word, "unknown"
    return None


def _parse_make(upper: str, original: str) -> str | None:
    m = _MAKE_RE.search(upper)
    return MAKES[m.group(1)] if m else None


def _parse_year(upper: str, original: str) -> int | None:
    m = YEAR_RE.search(_masked(upper))
    return int(m.group(1)) if m else None


@dataclass
class Vehicle:
    year: Hit | None = None
    make: Hit | None = None
    model: Hit | None = None


def find_vehicle(page: Page) -> Vehicle:
    v = Vehicle()
    # 1. Labeled fields ("Year: 2017  Make: Toyota  Model: RAV4").
    v.year = _first(find_labeled(page, _labels(r"(?<![A-Z/])(?:MODEL\s+)?(?:YEAR|YR)(?![A-Z/])(?!\s*/)"),
                                 _parse_year, base_conf=0.9))
    v.make = _first(find_labeled(page, _labels(r"(?<![A-Z/])MAKE(?![A-Z/])(?!\s*/)"), _parse_make, base_conf=0.9))
    if v.make:
        make = v.make.value

        def parse_labeled_model(upper: str, original: str):
            got = match_model(upper.lstrip(" :#-"), original.lstrip(" :#-"), make)
            return got[0] if got and (got[1] != "unknown" or upper.lstrip().startswith(":")) else None

        v.model = _first(find_labeled(page, _labels(r"(?<![A-Z/])MODEL(?![A-Z/])(?!\s*(?:YEAR|YR|/))"),
                                      parse_labeled_model, base_conf=0.85))

    # 2. Free-text descriptions ("2019 Chevy Silverado 1500 LT", "Ford F150 XLT 2018").
    best = None
    for li, line in enumerate(page.lines):
        masked = _masked(line.upper)
        for m in _MAKE_RE.finditer(line.upper):
            make = MAKES[m.group(1)]
            model = match_model(line.upper[m.end():], line.text[m.end():], make)
            years = [(abs(y.start() - m.start()), int(y.group(1))) for y in YEAR_RE.finditer(masked)]
            year = min(years)[1] if years else None
            known = bool(model and model[1] != "unknown")
            score = (2 if known else 0) + (1 if year else 0)
            if score == 0 or (not known and not year):
                continue
            if model and not known and not year:
                continue
            if best is None or score > best[0]:
                conf = min(line.segs[line.seg_index(m.start())].score, 1.0)
                best = (score, li, make, model, year, conf, line.text)
    if best:
        _, li, make, model, year, conf, text = best
        v.make = v.make or Hit(make, 1, li, 0.85 * conf, text)
        if model and (v.make.value == make):
            model_conf = {"known": 0.85, "ocr": 0.65, "unknown": 0.45}[model[1]]
            v.model = v.model or Hit(model[0], 1, li, model_conf * conf, text)
        if year:
            v.year = v.year or Hit(year, 1, li, 0.85 * conf, text)
    return v


# --- VIN-encoded year and make ----------------------------------------------------

_YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"  # 1980..2009, then repeats from 2010

WMI_MAKES = {
    "1HG": "Honda", "2HG": "Honda", "5FN": "Honda", "5J6": "Honda", "7FA": "Honda", "JHM": "Honda",
    "SHH": "Honda", "19X": "Honda", "2HK": "Honda", "19U": "Acura", "5J8": "Acura", "JH4": "Acura",
    "1G1": "Chevrolet", "2G1": "Chevrolet", "3G1": "Chevrolet", "1GC": "Chevrolet", "2GC": "Chevrolet",
    "3GC": "Chevrolet", "1GN": "Chevrolet", "2GN": "Chevrolet", "3GN": "Chevrolet", "KL7": "Chevrolet",
    "1GT": "GMC", "3GT": "GMC", "1GK": "GMC", "1GY": "Cadillac", "1G4": "Buick", "5GA": "Buick",
    "1FA": "Ford", "1FM": "Ford", "1FT": "Ford", "1FD": "Ford", "3FA": "Ford", "2FM": "Ford", "3FM": "Ford",
    "1LN": "Lincoln", "5LM": "Lincoln",
    "4T1": "Toyota", "4T3": "Toyota", "4T4": "Toyota", "5TD": "Toyota", "5TF": "Toyota", "2T1": "Toyota",
    "2T3": "Toyota", "3TM": "Toyota", "JTD": "Toyota", "JTE": "Toyota", "JTM": "Toyota", "JTN": "Toyota",
    "JTH": "Lexus", "2T2": "Lexus", "58A": "Lexus",
    "1N4": "Nissan", "1N6": "Nissan", "3N1": "Nissan", "5N1": "Nissan", "JN1": "Nissan", "JN8": "Nissan",
    "1J4": "Jeep", "1J8": "Jeep", "1C6": "Ram", "3C6": "Ram", "3D7": "Ram",
    "4S3": "Subaru", "4S4": "Subaru", "JF1": "Subaru", "JF2": "Subaru",
    "5NP": "Hyundai", "KMH": "Hyundai", "5NM": "Hyundai", "KNA": "Kia", "KND": "Kia", "5XY": "Kia", "5XX": "Kia",
    "5YJ": "Tesla", "7SA": "Tesla", "7G2": "Tesla",
    "3VW": "Volkswagen", "1VW": "Volkswagen", "WVW": "Volkswagen", "WVG": "Volkswagen",
    "WBA": "BMW", "WBS": "BMW", "5UX": "BMW", "5YM": "BMW", "JM1": "Mazda", "JM3": "Mazda",
    "WDD": "Mercedes-Benz", "WDC": "Mercedes-Benz", "4JG": "Mercedes-Benz", "55S": "Mercedes-Benz",
    "W1K": "Mercedes-Benz", "W1N": "Mercedes-Benz", "WAU": "Audi", "WA1": "Audi", "YV1": "Volvo", "YV4": "Volvo",
    "WP0": "Porsche", "WP1": "Porsche", "JA4": "Mitsubishi", "ML3": "Mitsubishi",
    "1G6": "Cadillac", "1GB": "Chevrolet", "3GK": "GMC", "2HN": "Acura", "3N6": "Nissan", "KM8": "Hyundai",
    "3KP": "Kia", "5TB": "Toyota", "5TE": "Toyota", "JTJ": "Lexus", "JTK": "Toyota", "3MZ": "Mazda",
    "WBX": "BMW", "4US": "BMW", "WMW": "Mini", "SAL": "Land Rover", "SAJ": "Jaguar", "2FA": "Ford",
    "1ZV": "Ford", "1B3": "Dodge", "2B3": "Dodge", "1D3": "Dodge",
    # Real WMIs shared by several makes (Chrysler/Dodge/Jeep/Ram): known, but no single make.
    "1C3": None, "1C4": None, "2C3": None, "2C4": None, "3C4": None, "1D7": None, "2D4": None,
}


def vin_model_year(v: str) -> int | None:
    if len(v) != 17 or v[9] not in _YEAR_CODES:
        return None
    base = 1980 + _YEAR_CODES.index(v[9])
    this_year = date.today().year
    if v[0] in "12345":  # North American rule: position 7 letter -> 2010+ cycle
        year = base + 30 if v[6].isalpha() else base
        return year if year <= this_year + 1 else base
    return base + 30 if base + 30 <= this_year + 1 else base


def vin_make(v: str) -> str | None:
    return WMI_MAKES.get(v[:3])


# --- Fields -----------------------------------------------------------------------

INVOICE_NO_LABELS = (
    (0, r"INVOICE\s*(?:NO\.?|NUMBER|NUM\.?|#)|(?<![A-Z])INV\.?\s*(?:NO\.?|#)"),
    (1, r"REPAIR\s*ORDER\s*(?:NO\.?|NUMBER|#)?|(?<![A-Z])R\.?\s?O\.?\s*(?:NO\.?|NUMBER|#)|(?<![A-Z])RO(?![A-Z])"
        r"|WORK\s*ORDER\s*(?:NO\.?|NUMBER|#)?|(?<![A-Z])W\.?O\.?\s*#|RECEIPT\s*(?:NO\.?|NUMBER|#)?"
        r"|STATEMENT\s*(?:NO\.?|NUMBER|#)|ORDER\s*(?:NO\.?|#)|(?<![A-Z])REF(?:ERENCE)?\.?\s*(?:NO\.?|NUMBER|#)?"
        r"|DOC(?:UMENT)?\s*(?:NO\.?|#)"),
    (2, r"(?<![A-Z])INVOICE(?![A-Z])(?!\s*(?:DATE|TOTAL|TO\b|AMOUNT))|(?<![A-Z])NO\.(?=\s*\S*\d)"),
)
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")


def parse_invoice_no(upper: str, original: str) -> str | None:
    if re.match(r"\s*:?\s*DATE", upper):
        return None
    if _PHONE_RE.search(upper):
        upper = _PHONE_RE.sub(lambda m: " " * len(m.group()), upper)
    dates = date_spans(upper)
    for m in re.finditer(r"(?<![A-Z0-9])([A-Z]{0,5}[-/ ]?\d[A-Z0-9\-/]{1,15})", upper):
        a, b = m.span(1)
        cand = m.group(1).strip(" -/")
        if any(da < b and a < db for da, db, _ in dates) or MONEY_RE.fullmatch(cand):
            continue
        if sum(c.isdigit() for c in cand) >= 3:
            return cand.lstrip(" ")
    return None


DATE_LABELS = (
    (0, r"INVOICE\s*DATE|INV\.?\s*DATE|DATE\s*OF\s*INVOICE|BILLING\s*DATE|STATEMENT\s*DATE"),
    (1, r"DATE\s*(?:OUT|CLOSED|COMPLETED)|(?<![A-Z])(?:CLOSED|COMPLETED|ISSUED)(?![A-Z])|SERVICE\s*DATE"
        r"|R\.?\s?O\.?\s*DATE|(?<!DUE )(?<!DUE)(?<![A-Z])DATE(?![A-Z])(?!\s*(?:IN|OUT|DUE|CLOSED|OF\s+BIRTH)\b)"),
    (2, r"DATE\s*IN(?![A-Z])"),
)

TOTAL_LABELS = (
    (0, r"TOTAL\s*(?:AMOUNT\s*)?DUE|AMOUNT\s*DUE|BALANCE\s*DUE|GRAND\s*TOTAL|INVOICE\s*TOTAL|TOTAL\s*AMOUNT"
        r"|AMOUNT\s*PAYABLE|TOTAL\s*INVOICE|PLEASE\s*PAY|TOTAL\s*CHARGES"),
    (1, r"(?<!SUB )(?<!SUB-)(?<![A-Z])TOTAL(?![A-Z])(?!\s*(?:PARTS|LABOU?R|TAX|HOURS|HRS|QTY|SUBLET|MATERIALS?"
        r"|DISCOUNT|SAVINGS)\b)|(?<![A-Z])BALANCE(?![A-Z])(?!\s*FORWARD)"),
)
_NOT_GRAND_TOTAL = re.compile(r"(?:PARTS|LABOU?R|TAX|SUBLET|MATERIALS?|PAINT)\s*$")

MILEAGE_LABELS = (
    (0, r"(?:MILEAGE|ODOMETER|ODO|MILES|MI\.?)\s*IN(?![A-Z])"),
    (1, r"(?<![A-Z])(?:MILEAGE|ODOMETER(?:\s*READING)?|ODO|MILES)(?![A-Z])(?!\s*(?:OUT|IN)(?![A-Z]))"),
    (2, r"(?<![A-Z0-9])[O0][D0][O0](?![A-Z0-9])(?=\s+\d)"),  # OCR: "ODO" read as 000 / 0D0
)

PLATE_LABELS = (
    (0, r"(?<![A-Z])LIC(?:ENSE)?\.?\s*(?:PLATE)?\s*(?:NO\.?|NUMBER|#)?(?![A-Z])"
        r"|(?<![A-Z])PLATE(?:\s*(?:NO\.?|NUMBER|#|/\s*ST(?:ATE)?))?(?![A-Z])"
        r"|(?<![A-Z])REG(?:ISTRATION)?\.?\s*(?:NO\.?|#)?(?![A-Z])"),
    (1, r"(?<![A-Z])TAG(?![A-Z])(?!\s*/\s*HAT)\s*(?:NO\.?|#)?"),
)

COLOR_LABELS = (
    (0, r"(?<![A-Z])(?:EXT(?:ERIOR)?\.?\s*)?COLOU?R(?![A-Z])|(?<![A-Z])CLR(?![A-Z])"),
    (1, r"(?<![A-Z])PAINT(?:\s*COLOU?R)?(?![A-Z])(?!\s*(?:MAT|MATERIALS?|SUPPLIES|LABOU?R|CODE)\b)"),
)

DOC_TITLE_RE = re.compile(
    r"(?:FINAL |TAX |SERVICE |CUSTOMER |ORIGINAL )?(?:INVOICE|REPAIR ORDER|RECEIPT|ESTIMATE|STATEMENT|WORK ORDER"
    r"|QUOTE|BILL|COPY|PAGE \d+(?: OF \d+)?)"
)


def _collect(page: Page, table, parse, base_conf=0.9, below=2) -> list[Hit]:
    hits = []
    for priority, pattern in table:
        conf = base_conf - 0.1 * priority
        hits += find_labeled(page, re.compile(pattern), parse, priority=priority, base_conf=conf, below=below)
    return hits


def _vendor(page: Page) -> Hit | None:
    candidates = []
    for li, line in enumerate(page.lines):
        if li >= 4 and line.y > 0.3 * page.height:
            break
        for s in line.segs:
            t = s.text.strip()
            up = re.sub(r"[^A-Z0-9 ]", "", t.upper()).strip()
            letters = sum(c.isalpha() for c in t)
            if (letters < 4 or ":" in t or "#" in t or t[0].isdigit() or t[0] in "($"
                    or _PHONE_RE.search(t) or "@" in t or "WWW" in up or parse_date(t)
                    or DOC_TITLE_RE.fullmatch(up)):
                continue
            candidates.append((-round(s.height, 1), li, t, s.score, line.text))
    if not candidates:
        return None
    _, li, t, score, text = min(candidates)
    return Hit(t, 0, li, 0.8 * score, text)


MIN_INVOICE_DATE = date(2000, 1, 1)
MAX_MILEAGE = 1_000_000


def implausible(name: str, value) -> str | None:
    """Why a field value can't be right, or None."""
    if name == "invoice_date" and isinstance(value, date):
        if value > date.today():
            return f"invoice_date {value} is in the future"
        if value < MIN_INVOICE_DATE:
            return f"invoice_date {value} is before 2000"
    if name == "mileage" and isinstance(value, int) and value >= MAX_MILEAGE:
        return f"mileage {value} is not under 1,000,000"
    if name == "total" and isinstance(value, Decimal) and value <= 0:
        return f"total {value} is not above zero"
    return None


@dataclass
class FieldValue:
    value: object
    confidence: float
    evidence: str = ""
    source: str = ""  # page source ("pdf_text", "ocr:paddle", "text"), "vin" (decoded) or "vision:<model>"


@dataclass
class VinConfirmation:
    """Independent evidence about the extracted VIN. Sources: "text_layer" (digital
    PDF), "vpic" (decode agrees with the invoice), "vision" (the vision model read the
    same VIN); later "photo_vin"."""

    source: str
    agrees: bool
    detail: str = ""


@dataclass
class InvoiceFields:
    fields: dict[str, FieldValue]
    vins: list[vinlib.VinResult]
    text_source: str
    text: str = field(repr=False, default="")
    vin_result: vinlib.VinResult | None = None  # the reading behind fields["vin"]
    multiple_vins: bool = False
    vin_unlabeled_only: bool = False
    vin_issues: list[str] = field(default_factory=list)  # set by _finalize
    other_reasons: list[str] = field(default_factory=list)
    vin_confirmations: list[VinConfirmation] = field(default_factory=list)
    vision_vin: str | None = None  # check-digit-valid VIN read by the vision model
    conflicts: dict[str, dict] = field(default_factory=dict)  # field -> {"rules": ..., "vision": ...}
    notes: list[str] = field(default_factory=list)  # informational, no review

    @property
    def vin_trusted(self) -> bool:
        """May the VIN auto-link? No VIN issue and at least one agreeing confirmation."""
        return ("vin" in self.fields and not self.vin_issues
                and any(c.agrees for c in self.vin_confirmations))

    @property
    def plate_trusted(self) -> bool:
        """May matching auto-link on the plate? Not if only the vision model read it."""
        return "plate" in self.fields and not self.fields["plate"].source.startswith("vision:")

    @property
    def review_reasons(self) -> list[str]:
        reasons = self.vin_issues + self.other_reasons
        reasons += [why for name, fv in self.fields.items() if (why := implausible(name, fv.value))]
        reasons += [f"{name}: OCR read {c['rules']}, vision model read {c['vision']}"
                    for name, c in self.conflicts.items()]
        if "vin" in self.fields and not self.vin_issues and not self.vin_trusted:
            why = "; ".join(f"{c.source}: {c.detail}" for c in self.vin_confirmations) or "no confirmation source"
            reasons.append(f"VIN read by OCR not confirmed ({why})")
        return reasons

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def add_vin_confirmation(self, confirmation: VinConfirmation) -> None:
        self.vin_confirmations.append(confirmation)

    def value(self, name: str):
        fv = self.fields.get(name)
        return fv.value if fv else None

    def to_dict(self) -> dict:
        return {
            "fields": {k: {"value": str(v.value), "confidence": round(v.confidence, 3), "source": v.source}
                       for k, v in self.fields.items()},
            "vin_trusted": self.vin_trusted,
            "plate_trusted": self.plate_trusted,
            "conflicts": self.conflicts,
            "vin_confirmations": [asdict(c) for c in self.vin_confirmations],
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "text_source": self.text_source,
            "notes": self.notes,
        }


def _put(out: dict, name: str, hit: Hit | None, source: str) -> None:
    if hit is not None:
        out[name] = FieldValue(hit.value, max(min(hit.confidence, 1.0), 0.01), hit.evidence, source)


def _extract_fields(page: Page) -> InvoiceFields:
    """Fields as read from one page; review and trust are decided in _finalize."""
    out: dict[str, FieldValue] = {}
    src = page.source

    # VIN
    results = vinlib.extract_vins(page.text)
    usable = [r for r in results if r.status is not vinlib.VinStatus.INVALID]
    labeled = [r for r in usable if not r.unlabeled_assembly]
    pool = labeled or usable
    best = max(pool, key=lambda r: r.confidence) if pool else None
    if best is not None and best.vin:
        out["vin"] = FieldValue(best.vin, best.confidence, best.raw, src)

    # Plate
    plate = _first(_collect(page, PLATE_LABELS, lambda u, o: parse_plate(u)))
    if plate:
        p, state, guessed = plate.value
        _put(out, "plate", Hit(p, 0, plate.line, plate.confidence, plate.evidence), src)
        if state:
            conf = plate.confidence * (0.6 if guessed else 1.0)
            _put(out, "plate_state", Hit(state, 0, plate.line, conf, plate.evidence), src)

    # Vehicle (VIN fallback happens in _finalize, after any OCR fill)
    veh = find_vehicle(page)
    _put(out, "year", veh.year, src)
    _put(out, "make", veh.make, src)
    _put(out, "model", veh.model, src)

    _put(out, "color", _first(_collect(page, COLOR_LABELS, lambda u, o: normalize_color(u))), src)
    _put(out, "mileage", _first(_collect(page, MILEAGE_LABELS, lambda u, o: parse_mileage(u))), src)

    # Date: labeled, else the first date near the top of the page
    dt = _first(_collect(page, DATE_LABELS, lambda u, o: parse_date(u)))
    if dt is None:
        top = [(li, line) for li, line in enumerate(page.lines) if li < 4 or line.y <= 0.4 * page.height]
        for li, line in top:
            spans = [s for s in date_spans(line.upper) if not re.search(r"(?<![A-Z])DUE(?![A-Z])", line.upper[:s[0]])]
            if spans:
                dt = Hit(spans[0][2], 9, li, 0.5 * line.segs[0].score, line.text)
                break
    _put(out, "invoice_date", dt, src)

    # Total: strongest label, lowest on the page
    def parse_total(upper: str, original: str):
        values = _money_values(upper)
        return values[-1] if values else None

    totals = [h for h in _collect(page, TOTAL_LABELS, parse_total, below=1)
              if not _NOT_GRAND_TOTAL.search(page.lines[h.line].upper[: page.lines[h.line].upper.find("TOTAL")])
              or "TOTAL" not in page.lines[h.line].upper]
    _put(out, "total", min(totals, key=lambda h: (h.priority, -h.line), default=None), src)

    _put(out, "invoice_no", _first(_collect(page, INVOICE_NO_LABELS, parse_invoice_no, base_conf=0.85)), src)
    _put(out, "vendor", _vendor(page), src)

    return InvoiceFields(
        out, results, src, page.text, vin_result=best if best is not None and best.vin else None,
        multiple_vins=len({r.vin or r.alternatives for r in labeled}) > 1,
        vin_unlabeled_only=bool(usable) and not labeled,
    )


def _fill_missing(f: InvoiceFields, g: InvoiceFields) -> None:
    """Fill fields missing in f (text layer) from g (OCR of the same page). Existing
    values are never overwritten."""
    for name in FIELD_NAMES:
        if name not in f.fields and name in g.fields:
            if name == "plate_state" and f.fields.get("plate") is not g.fields.get("plate"):
                continue  # a state only makes sense with the plate it was read with
            f.fields[name] = g.fields[name]
    if f.fields.get("vin") is g.fields.get("vin") and "vin" in g.fields:
        f.vins, f.vin_result = g.vins, g.vin_result
        f.multiple_vins, f.vin_unlabeled_only = g.multiple_vins, g.vin_unlabeled_only
    f.text_source = f"{f.text_source}+{g.text_source}"


VpicLookup = Callable[[str], vinlib.VpicDecode]


def _finalize(f: InvoiceFields, vpic: VpicLookup | None | bool = None) -> InvoiceFields:
    """Decide VIN issues, review reasons and VIN confirmations. ``vpic``: a lookup
    function, None for the real vPIC API, or False to skip vPIC."""
    S = vinlib.VinStatus
    issues: list[str] = []
    r = f.vin_result
    if f.multiple_vins:
        issues.append("multiple VINs on invoice")
    if r is not None:
        if f.vin_unlabeled_only:
            issues.append("VIN only found in split/glued text without a VIN label")
        elif r.status is S.UNVERIFIED:
            issues.append("VIN fails the check digit")
        elif r.status is S.AMBIGUOUS:
            issues.append("VIN has several possible OCR repairs")
        elif r.status is S.REPAIRED:
            issues.append("VIN was repaired from a likely OCR misread")
        if r.vin and r.vin[:3] not in WMI_MAKES:
            issues.append(f"unknown manufacturer code (WMI) {r.vin[:3]}")

    vin_value = r.vin if r is not None and r.vin and not f.vin_unlabeled_only else None
    vin_source = f.fields["vin"].source if "vin" in f.fields else ""
    if vin_value and vin_source.startswith("vision:"):
        issues.append("VIN read only by the vision model")
    elif vin_value and vin_source.startswith("ocr:") and f.vision_vin and f.vision_vin != vin_value:
        issues.append(f"vision model read a different VIN ({f.vision_vin})")
    if vin_value and r.status in (S.VALID, S.NORMALIZED):
        for name, decoded in (("year", vin_model_year(vin_value)), ("make", vin_make(vin_value))):
            if decoded is None:
                continue
            if name not in f.fields:
                f.fields[name] = FieldValue(decoded, 0.6, f"VIN {vin_value}", "vin")
            elif f.fields[name].value != decoded:
                issues.append(f"{name} {f.fields[name].value} disagrees with VIN ({decoded})")
    f.vin_issues = issues
    f.other_reasons = [r for r in f.other_reasons if r != "no VIN or plate found"]
    if "vin" not in f.fields and "plate" not in f.fields:
        f.other_reasons.append("no VIN or plate found")

    f.vin_confirmations = []
    if vin_value and not issues:
        if not vin_source.startswith("ocr:"):
            f.add_vin_confirmation(VinConfirmation("text_layer", True, vin_source))
        else:
            f.add_vin_confirmation(_vpic_confirmation(f, vin_value, vpic))
            if f.vision_vin == vin_value:
                f.add_vin_confirmation(VinConfirmation("vision", True, "vision model read the same VIN"))
    return f


def _printed(f: InvoiceFields, name: str):
    """A value the rules read on the invoice (not decoded from the VIN, not from the vision model)."""
    fv = f.fields.get(name)
    return fv.value if fv is not None and fv.source != "vin" and not fv.source.startswith("vision:") else None


def _vpic_confirmation(f: InvoiceFields, vin_value: str, vpic) -> VinConfirmation:
    """Clean vPIC decode (no problem error codes; see vin.VPIC_INFORMATIONAL_CODES) that
    agrees with the year and make *printed* on the invoice, and the model when both
    have one. A partial decode (code 14) must also agree on the model. Values decoded
    from the VIN itself don't count."""
    year, make, model = _printed(f, "year"), _printed(f, "make"), _printed(f, "model")
    if vpic is False:
        return VinConfirmation("vpic", False, "vPIC check disabled")
    if year is None or make is None:
        return VinConfirmation("vpic", False, "invoice has no printed year and make to compare")
    d = (vpic or vinlib.decode_vpic)(vin_value)
    if not d.ok:
        return VinConfirmation("vpic", False, d.error_text or "decode failed")
    if not d.clean:
        return VinConfirmation("vpic", False, f"VIN problem codes {','.join(d.problem_codes)}: {d.error_text[:100]}")
    if d.partial and (model is None or d.model is None):
        return VinConfirmation("vpic", False, "partial decode (code 14) and no model to compare")
    problems = vinlib.compare_with_invoice(d, year=year, make=make, model=model)
    if problems or not vinlib.agrees_with_invoice(d, year=year, make=make, model=model):
        return VinConfirmation("vpic", False, "; ".join(problems) or "no agreement")
    return VinConfirmation("vpic", True, f"{d.year} {d.make} {d.model or ''}".strip())


def extract_from_page(page: Page, vpic: VpicLookup | None | bool = None) -> InvoiceFields:
    return _finalize(_extract_fields(page), vpic)


def extract_invoice(path: str | Path, ocr: OcrEngine | None = None, cache_dir: str | Path | None = None,
                    vpic: VpicLookup | None | bool = None, vision=None) -> InvoiceFields:
    """Extract one invoice. For a PDF with a text layer, any field missing from it is
    filled from OCR of the rendered page. With ``vision`` (app.extract.vision.VisionModel)
    remaining gaps go to the vision model; the OCR worker is stopped first and the
    model unloaded after. For batches, see tools/accuracy.py: rules for every file,
    then ``stop_ocr()`` and one ``vision.session()``."""
    f = _extract_rules(path, ocr, cache_dir)
    if vision is None:
        return _finalize(f, vpic)
    called = []

    def before_model():
        stop_ocr(ocr)
        called.append(True)

    try:
        return fill_gaps_with_vision(f, path, vision, vpic=vpic, before_model=before_model)
    finally:
        if called:
            vision.unload()


def _extract_rules(path, ocr, cache_dir) -> InvoiceFields:
    """Fields from the text layer and/or OCR, before _finalize."""
    try:
        page = read_page(path, ocr=ocr, cache_dir=cache_dir)
    except OcrFailed as e:  # unreadable: no fields, straight to review
        return InvoiceFields(fields={}, vins=[], text_source="ocr_failed", other_reasons=[_ocr_failure(e)])
    f = _extract_fields(page)
    if page.source == "pdf_text" and any(name not in f.fields for name in FIELD_NAMES):
        try:
            ocr_page = read_page(path, ocr=ocr, cache_dir=cache_dir, force_ocr=True)
            _fill_missing(f, _extract_fields(ocr_page))
        except OcrFailed as e:  # keep the text layer; missing fields stay missing
            f.other_reasons.append(_ocr_failure(e))
    return f


def _ocr_failure(e: OcrFailed) -> str:
    what = "OCR timed out" if isinstance(e, OcrTimeout) else "OCR failed"
    return f"{what} ({e}); image treated as unreadable"


# --- Vision model: fill what the rules left -----------------------------------------

VISION_FILL_BELOW = 0.5  # rule values under this confidence may be replaced by the model
VISION_CONFIDENCE = 0.5  # confidence given to values taken from the model
VISION_MAX_PAGES = 2
# No check digit to say which reading is right: a disagreement goes to review.
UNCHECKED_FIELDS = frozenset({"invoice_date", "invoice_no", "total", "mileage"})


def stop_ocr(ocr: OcrEngine | None = None) -> None:
    """Stop the OCR worker process (``ocr``, or the shared engine if one was started) so
    its memory is free before a vision model loads. It restarts on the next OCR call.
    An in-process engine (OCR_TIMEOUT=0) has no worker to stop."""
    if ocr is None and get_ocr_engine.cache_info().currsize:
        ocr = get_ocr_engine()
    close = getattr(ocr, "close", None)
    if close is not None:
        close()


def _vision_images(path: Path) -> list:
    if path.suffix.lower() == ".pdf":
        return _rasterize_pdf(path)[:VISION_MAX_PAGES]
    return [_load_image(path)]


def _vision_value(name: str, raw: str, make: str | None):
    """Parse one model answer with the rules' own parsers; None if it doesn't parse."""
    raw = raw.strip()
    upper = raw.upper()
    if not raw or upper in {"NULL", "NONE", "N/A", "UNKNOWN"}:
        return None
    if name == "plate":
        p = re.sub(r"[^A-Z0-9]", "", upper)
        return p if 2 <= len(p) <= 8 and any(c.isdigit() for c in p) else None
    if name == "plate_state":
        return upper if upper in US_STATES else None
    if name == "year":
        return int(raw) if YEAR_RE.fullmatch(raw) else None
    if name == "make":
        return _parse_make(upper, raw)
    if name == "model":
        m = match_model(upper, raw, make) if make else None
        return m[0] if m and m[1] == "known" else None  # catalog models only
    if name == "color":
        return normalize_color(raw)
    if name == "mileage":
        return parse_mileage(upper)
    if name == "invoice_date":
        return parse_date(raw)
    if name == "total":
        values = _money_values(upper)
        if not values:
            return None
        negative = re.match(r"\s*\(?\s*-|.*\d\s*-\s*$", raw)  # "-45.00", "$45.00-"
        return -values[-1] if negative else values[-1]
    if name == "invoice_no":
        return parse_invoice_no(upper, raw)
    if name == "vendor":
        return raw if len(raw) >= 3 and sum(c.isalpha() for c in raw) >= 3 else None
    return None


# Looser than PLATE_LABELS: only a word start, so OCR-garbled labels ("LICA" for "LIC#",
# "P1ATE") still count. The plate itself must also be nearby (see _plate_near_label).
_PLATE_LABEL_RE = re.compile(r"(?<![A-Z])(?:LIC|P[L1I]AT|TAG|REG)")
_PLATE_FOLD = str.maketrans("IOLSBZGDQ", "101582600")


def _plate_near_label(text: str, plate: str) -> bool:
    """Does the OCR text have a plate label (Plate, Lic, License, Reg, Tag, even garbled) with something
    like ``plate`` after it on the same line or within the next two lines? "Like" = a
    common run of at least half the plate (min. 3 characters) after folding OCR look-alikes."""
    from difflib import SequenceMatcher

    lines = text.upper().split("\n")
    want = plate.translate(_PLATE_FOLD)
    need = max(3, math.ceil(len(want) / 2))
    for i, line in enumerate(lines):
        m = _PLATE_LABEL_RE.search(line)
        if not m:
            continue
        window = re.sub(r"[^A-Z0-9]", "", " ".join([line[m.end():], *lines[i + 1:i + 3]])).translate(_PLATE_FOLD)
        match = SequenceMatcher(None, window, want, autojunk=False).find_longest_match(0, len(window), 0, len(want))
        if match.size >= min(need, len(want)):
            return True
    return False


def _ocr_read_vin(f: InvoiceFields, v: str) -> bool:
    """Is ``v`` one of the readings OCR produced (best, alternatives, other candidates)?"""
    return any(r.vin == v or v in r.alternatives for r in f.vins)


def _vision_vin(f: InvoiceFields, raw: str, name: str, vpic) -> None:
    v = vinlib.normalize(raw)
    if not vinlib.is_valid(v):  # 17 chars, VIN alphabet, check digit
        return
    f.vision_vin = v
    cur = f.fields.get("vin")
    weak_ocr = cur is not None and cur.source.startswith("ocr:") and (
        f.vin_unlabeled_only or f.vin_result is None
        or f.vin_result.status not in (vinlib.VinStatus.VALID, vinlib.VinStatus.NORMALIZED))
    if cur is not None and not weak_ocr:
        return  # _finalize compares it with the rules' VIN
    if not (_ocr_read_vin(f, v) or _vpic_confirmation(f, v, vpic).agrees):
        return
    f.fields["vin"] = FieldValue(v, VISION_CONFIDENCE, raw, name)
    f.vin_result = vinlib.repair(v)
    f.vin_unlabeled_only = False


def fill_gaps_with_vision(f: InvoiceFields, path: str | Path, vision, vpic: VpicLookup | None | bool = None,
                          before_model: Callable[[], None] | None = None,
                          fill_below: float = VISION_FILL_BELOW) -> InvoiceFields:
    """Ask ``vision`` for the fields the rules left blank or below ``fill_below``
    confidence (and for the VIN when it isn't trusted), then re-run _finalize.
    ``before_model`` runs right before the model is called (used to stop OCR)."""
    f = _finalize(f, vpic)
    targets = {n for n, fv in ((n, f.fields.get(n)) for n in FIELD_NAMES)
               if fv is None or fv.confidence < fill_below or implausible(n, fv.value)}
    vin_source = f.fields["vin"].source if "vin" in f.fields else ""
    want_vin = "vin" in targets or (vin_source.startswith("ocr:") and not f.vin_trusted)
    if not targets and not want_vin:
        return f
    if before_model is not None:
        before_model()
    from app.extract.vision import VisionError

    try:
        raw = vision.read_invoice(_vision_images(Path(path)))
    except VisionError as e:
        f.notes.append(f"vision model failed: {e}")
        return f

    if raw.get("vin"):
        _vision_vin(f, raw["vin"], vision.name, vpic)
    for name in FIELD_NAMES:
        if name in ("vin", "plate_state") or name not in targets or not raw.get(name):
            continue
        make = f.value("make") if name == "model" else None
        value = _vision_value(name, raw[name], make)
        if value is None or implausible(name, value) or f.value(name) == value:
            continue
        cur = f.fields.get(name)
        if name in UNCHECKED_FIELDS and cur is not None and not cur.source.startswith("vision:"):
            f.conflicts[name] = {"rules": str(cur.value), "vision": raw[name]}
            continue
        if name == "plate" and not _plate_near_label(f.text, value):
            continue
        f.fields[name] = FieldValue(value, VISION_CONFIDENCE, raw[name], vision.name)
        if name == "plate" and "plate_state" in f.fields:
            del f.fields["plate_state"]  # it belonged to the plate that was replaced
    # A state only goes with the plate it was read with.
    plate = f.fields.get("plate")
    state = _vision_value("plate_state", raw.get("plate_state") or "", None)
    if state and plate is not None and plate.source == vision.name and "plate_state" not in f.fields:
        f.fields["plate_state"] = FieldValue(state, VISION_CONFIDENCE, raw["plate_state"], vision.name)
    return _finalize(f, vpic)
