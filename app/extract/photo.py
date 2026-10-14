"""Vehicle photo -> PhotoFields (shot type, VIN, plate, odometer, color, body style, EXIF).

Pipeline (``extract_photos`` works on a batch so each heavy engine loads once)
--------
0. Cheap, for every photo: EXIF (capture time, GPS, device) and barcodes (zxing-cpp).
1. Vision model (optional, OLLAMA_MODEL): shot type, plus its own reading of VIN,
   plate, odometer, color and body style. One request per photo.
2. OCR, only on VIN-plate, door-jamb and odometer shots (and not for a VIN the
   barcode already gave).
3. fast-alpr, only on plate and exterior shots.
Without a vision model every photo goes through OCR and ALPR and the shot type comes
from rules (barcode -> door jamb, a VIN in the text -> VIN plate, an odometer reading
-> odometer, a plate filling the frame -> plate, a smaller plate -> exterior).

Only one of OCR, ALPR and the vision model is loaded at a time (``Engines``): an 8 GB
machine can't hold them together.

Trust (same rules as invoices)
-----
VIN: a barcode VIN with a valid check digit is trusted ("barcode"). An OCR VIN is
trusted only with no issue (check digit, repairs, unknown WMI) and a second reader that
agrees: the vision model ("vision"). The model's VIN must pass the check digit; if it
differs from the OCR VIN, trust is blocked. A VIN only the model read is kept if it
matches an OCR reading or vPIC decodes it cleanly, and is never trusted.
Plate: plates have no check digit, so like an OCR VIN, fast-alpr's plate is trusted only
when a second reader (the model) read the same plate; fast-alpr's confidence is no guard
against O/0 swaps. A disagreement goes to review; a plate only the model read is never trusted.
Odometer: OCR and model disagreeing goes to review; the model may fill a blank.
Color and body style come from the model, on exterior shots only.
Capture time: EXIF, else the upload time (lower confidence).
"""

from __future__ import annotations

import gc
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from app.extract import vin as vinlib
from app.extract.invoice import (
    MAX_MILEAGE, US_STATES, WMI_MAKES, FieldValue, OcrFailed, Page, VinConfirmation, _load_image,
    get_ocr_engine, normalize_color, stop_ocr,
)

SHOT_TYPES = ("exterior", "vin_plate", "door_jamb", "odometer", "plate", "interior", "other")
VIN_SHOTS = frozenset({"vin_plate", "door_jamb"})
TEXT_SHOTS = VIN_SHOTS | {"odometer"}
# The model called flat dashboard VIN plates "other": OCR still looks at these, and a short
# text with a VIN overrides the model (invoice photos have 20+ lines, stickers about 9).
RECHECK_SHOTS = frozenset({"other", "interior"})
RECHECK_MAX_LINES = 12
ALPR_SHOTS = frozenset({"plate", "exterior"})
BODY_STYLES = ("sedan", "suv", "pickup", "hatchback", "coupe", "wagon", "van", "minivan", "convertible")

EXIF_CONFIDENCE = 0.9
UPLOAD_CONFIDENCE = 0.3
VISION_CONFIDENCE = 0.5
RULES_SHOT_CONFIDENCE = 0.4
BARCODE_CONFIDENCE = 0.99
PLATE_FILLS_FRAME = 0.05  # plate box / image area at or above this: a plate close-up
# Both readers agreed on "OFYD327" for "0FYD327": agreement can't settle O/0, I/1, Q/0.
CONFUSABLE_PLATE = re.compile(r"[OIQ]")


# --- Engines -------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


class FastAlpr:
    """fast-alpr (plate detector + plate OCR, ONNX), loaded on first use, freed by close()."""

    def __init__(self, detector_model: str = "yolo-v9-t-384-license-plate-end2end",
                 ocr_model: str = "cct-xs-v2-global-model") -> None:
        self.detector_model, self.ocr_model = detector_model, ocr_model
        self._alpr = None

    def __call__(self, image) -> list[PlateRead]:
        import numpy as np

        if self._alpr is None:
            from fast_alpr import ALPR

            self._alpr = ALPR(detector_model=self.detector_model, ocr_model=self.ocr_model)
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        out = []
        for r in self._alpr.predict(bgr):
            if r.ocr is None or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            conf = sum(conf) / len(conf) if isinstance(conf, list) and conf else float(conf or 0)
            b = r.detection.bounding_box
            out.append(PlateRead(r.ocr.text, conf, (b.x1, b.y1, b.x2, b.y2)))
        return out

    def close(self) -> None:
        self._alpr = None
        gc.collect()


class _SharedOcr:
    """The shared OCR engine from invoice.get_ocr_engine, started on first use."""

    @property
    def name(self) -> str:
        return get_ocr_engine().name

    @property
    def word_level(self) -> bool:
        return bool(getattr(get_ocr_engine(), "word_level", False))

    def __call__(self, image):
        return get_ocr_engine()(image)

    def close(self) -> None:
        stop_ocr()


class Engines:
    """OCR, ALPR and the vision model, at most one of them loaded at a time."""

    def __init__(self, ocr=None, alpr=None, vision=None) -> None:
        self.ocr = ocr if ocr is not None else _SharedOcr()
        self.alpr = alpr if alpr is not None else FastAlpr()
        self.vision = vision

    @classmethod
    def from_env(cls) -> Engines:
        from app.extract.vision import from_env

        return cls(vision=from_env())

    def _stop(self, name: str) -> None:
        if name == "ocr":
            stop_ocr(self.ocr)
        elif name == "alpr":
            self.alpr.close()
        elif name == "vision" and self.vision is not None:
            self.vision.unload()

    def activate(self, name: str) -> None:
        """Stop the other engines before ``name`` loads."""
        for other in ("ocr", "alpr", "vision"):
            if other != name:
                self._stop(other)

    def release(self) -> None:
        for name in ("ocr", "alpr", "vision"):
            self._stop(name)


# --- Readers -------------------------------------------------------------------------


@dataclass
class Exif:
    captured_at: datetime | None = None
    gps: tuple[float, float] | None = None
    device: str | None = None


def read_exif(path: str | Path) -> Exif:
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    try:
        with Image.open(path) as im:
            ex = im.getexif()
    except OSError:
        return Exif()
    out = Exif()
    raw = ex.get_ifd(0x8769).get(0x9003) or ex.get(0x0132)  # DateTimeOriginal, else DateTime
    if raw:
        try:
            out.captured_at = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    g = ex.get_ifd(0x8825)
    try:
        if g.get(2) and g.get(4):
            lat = _degrees(g[2]) * (-1 if str(g.get(1, "N")).upper().startswith("S") else 1)
            lon = _degrees(g[4]) * (-1 if str(g.get(3, "E")).upper().startswith("W") else 1)
            out.gps = (lat, lon)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    device = " ".join(str(ex.get(t, "")).strip("\x00 ") for t in (0x010F, 0x0110)).strip()
    out.device = device or None
    return out


def _degrees(dms) -> float:
    d, m, s = (float(x) for x in dms)
    return d + m / 60 + s / 3600


def read_barcode_vin(image) -> str | None:
    """A VIN from a barcode (door-jamb stickers carry Code 39 / Code 128 / Data Matrix).
    Some add an "I" (import) prefix. Only a VIN with a valid check digit counts."""
    import zxingcpp

    for code in zxingcpp.read_barcodes(image):
        text = vinlib.normalize(code.text or "")
        for cand in (text, text[1:] if len(text) == 18 and text[0] == "I" else ""):
            if len(cand) == 17 and vinlib.is_valid(cand):
                return cand
    return None


_ODO_NUM = re.compile(r"(?<![\d.,:])(\d{1,3}(?:,\d{3})+|\d{1,7})(?![\d:]|[.,]\d)")


def parse_odometer(text: str) -> int | None:
    """Total mileage from odometer-display text. Trip meters (TRIP lines, decimals),
    clock times and temperatures are skipped. A number needs a cue: after "ODO" on its
    line or on the line above, followed by mi/miles/km, or alone on its line."""
    lines = text.upper().split("\n")
    best: tuple[int, int] | None = None  # (priority, value)
    for i, line in enumerate(lines):
        if "TRIP" in line:
            continue
        odo = line.find("ODO")
        prev = lines[i - 1] if i else ""
        odo_above = "ODO" in prev and "TRIP" not in prev and not _ODO_NUM.search(prev[prev.find("ODO"):])
        for m in _ODO_NUM.finditer(line):
            tail = line[m.end():]
            if re.match(r"\s*(?:°|AM\b|PM\b)", tail):
                continue
            if (odo >= 0 and m.start() > odo) or odo_above:
                pri = 0
            elif re.match(r"\s*(?:MI|MILES|KM)\b", tail):
                pri = 1
            elif line.strip() == m.group(0) and len(m.group(0).replace(",", "")) >= 3:
                pri = 2
            else:
                continue
            value = int(m.group(1).replace(",", ""))
            if 0 < value < MAX_MILEAGE and (best is None or pri < best[0]):
                best = (pri, value)
    return best[1] if best else None


def _model_odometer(raw: str | None) -> int | None:
    m = re.fullmatch(r"\s*(\d{1,3}(?:,\d{3})+|\d{1,7})\s*(?:MI|MILES|KM)?\.?\s*", (raw or "").upper())
    if not m:
        return None
    value = int(m.group(1).replace(",", ""))
    return value if 0 < value < MAX_MILEAGE else None


def normalize_shot_type(raw: str | None) -> str:
    s = (raw or "").lower()
    if s in SHOT_TYPES:
        return s
    for words, shot in ((("jamb", "sticker", "label"), "door_jamb"), (("vin",), "vin_plate"),
                        (("odometer", "odo", "cluster", "gauge"), "odometer"), (("plate", "license", "tag"), "plate"),
                        (("interior", "seat", "cabin"), "interior"), (("exterior", "car", "vehicle"), "exterior")):
        if any(w in s for w in words):
            return shot
    return "other"


def normalize_body_style(raw: str | None) -> str | None:
    s = (raw or "").lower()
    for words, style in ((("minivan",), "minivan"), (("pickup", "truck"), "pickup"),
                         (("suv", "crossover", "sport utility"), "suv"), (("sedan", "saloon"), "sedan"),
                         (("hatch",), "hatchback"), (("wagon", "estate"), "wagon"), (("coupe",), "coupe"),
                         (("convertible", "cabrio", "roadster"), "convertible"), (("van",), "van")):
        if any(w in s for w in words):
            return style
    return None


def _plate_text(raw: str | None) -> str | None:
    p = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    return p if 2 <= len(p) <= 8 else None


# --- Result --------------------------------------------------------------------------


@dataclass
class PhotoFields:
    path: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    vin_issues: list[str] = field(default_factory=list)
    vin_confirmations: list[VinConfirmation] = field(default_factory=list)
    plate_confirmed: bool = False  # the model read the same plate as fast-alpr
    conflicts: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def value(self, name: str):
        fv = self.fields.get(name)
        return fv.value if fv else None

    @property
    def vin_trusted(self) -> bool:
        return "vin" in self.fields and not self.vin_issues and any(c.agrees for c in self.vin_confirmations)

    @property
    def plate_issues(self) -> list[str]:
        """Why matching may not auto-link on this plate (empty: trusted)."""
        fv = self.fields.get("plate")
        if fv is None:
            return []
        if fv.source != "alpr":
            return ["plate read only by the vision model"]
        if "plate" in self.conflicts:
            return []  # reported as a conflict
        issues = []
        if not self.plate_confirmed:
            issues.append("plate read by ALPR not confirmed by a second reader")
        if self.value("shot_type") != "plate":
            issues.append("plate read from a photo that is not a plate close-up")
        if CONFUSABLE_PLATE.search(fv.value):
            issues.append("plate has O, I or Q, easily confused with 0 and 1")
        return issues

    @property
    def plate_trusted(self) -> bool:
        """May matching auto-link on the plate? Only fast-alpr's on a close-up, confirmed by the
        model, undisputed and free of O/I/Q."""
        return "plate" in self.fields and "plate" not in self.conflicts and not self.plate_issues

    @property
    def review_reasons(self) -> list[str]:
        reasons = list(self.vin_issues)
        if "vin" in self.fields and not self.vin_issues and not self.vin_trusted:
            reasons.append("VIN read by OCR not confirmed by a second reader")
        reasons += [x for x in self.plate_issues if x != "plate read only by the vision model"]
        reasons += [f"{name}: " + ", ".join(f"{k} read {v}" for k, v in c.items()) for name, c in self.conflicts.items()]
        return reasons

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "fields": {k: {"value": str(v.value), "confidence": round(v.confidence, 3), "source": v.source}
                       for k, v in self.fields.items()},
            "vin_trusted": self.vin_trusted,
            "plate_trusted": self.plate_trusted,
            "vin_confirmations": [asdict(c) for c in self.vin_confirmations],
            "conflicts": self.conflicts,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "notes": self.notes,
        }


@dataclass
class _Item:
    path: Path
    uploaded_at: datetime | None
    image: object = None
    exif: Exif = field(default_factory=Exif)
    barcode_vin: str | None = None
    answer: dict | None = None  # vision model; None = not asked or failed
    shot: str | None = None
    shot_source: str = ""
    text: str | None = None  # OCR
    ocr_source: str = ""
    plates: list[PlateRead] | None = None  # fast-alpr
    notes: list[str] = field(default_factory=list)


# --- Pipeline ------------------------------------------------------------------------


def extract_photo(path: str | Path, engines: Engines, uploaded_at: datetime | None = None,
                  vpic=None) -> PhotoFields:
    return extract_photos([path], engines, uploaded_at=uploaded_at, vpic=vpic)[0]


def extract_photos(paths, engines: Engines, uploaded_at: datetime | dict | None = None,
                   vpic=None, progress=None) -> list[PhotoFields]:
    """``uploaded_at``: one time for all photos or {path: time}. ``vpic``: lookup function,
    None for the real API, False to skip (then a model-only VIN is never kept).
    ``progress(step)`` is called with "vision", "ocr" and "alpr" before each pass."""
    items = []
    for p in paths:
        up = uploaded_at.get(p, uploaded_at.get(Path(p))) if isinstance(uploaded_at, dict) else uploaded_at
        it = _Item(Path(p), up)
        it.image = _load_image(it.path)
        it.exif = read_exif(it.path)
        it.barcode_vin = read_barcode_vin(it.image)
        items.append(it)
    try:
        for step, run in (("vision", _vision_pass), ("ocr", _ocr_pass), ("alpr", _alpr_pass)):
            if progress is not None:
                progress(step)
            run(items, engines)
    finally:
        engines.release()
    vision_name = getattr(engines.vision, "name", "vision")
    return [_combine(it, vision_name, vpic) for it in items]


def _vision_pass(items: list[_Item], engines: Engines) -> None:
    if engines.vision is None:
        return
    from app.extract.vision import VisionError

    engines.activate("vision")
    with engines.vision.session():
        for it in items:
            try:
                it.answer = engines.vision.read_photo([it.image])
            except VisionError as e:
                it.notes.append(f"vision model failed: {e}")
                continue
            it.shot, it.shot_source = normalize_shot_type(it.answer.get("shot_type")), engines.vision.name


def _ocr_pass(items: list[_Item], engines: Engines) -> None:
    todo = [it for it in items if (it.shot in TEXT_SHOTS | RECHECK_SHOTS if it.shot else True)
            and not (it.barcode_vin and it.shot in (VIN_SHOTS | {None}))]
    if not todo:
        return
    engines.activate("ocr")
    for it in todo:
        try:
            segs = engines.ocr(it.image)
        except OcrFailed as e:
            it.notes.append(f"OCR failed: {e}")
            continue
        page = Page.from_segs(segs, it.image.width, it.image.height, f"ocr:{engines.ocr.name}",
                              merge_words=bool(getattr(engines.ocr, "word_level", False)))
        it.text, it.ocr_source = page.text, page.source
    for it in items:  # rules shot type from barcode and text
        if it.shot in RECHECK_SHOTS and it.text and it.text.count("\n") < RECHECK_MAX_LINES \
                and any(r.vin for r in vinlib.extract_vins(it.text)):
            it.notes.append(f"vision model said {it.shot}; OCR found a VIN")
            it.shot = None
        if it.shot is None:
            if it.barcode_vin:
                it.shot = "door_jamb"
            elif it.text and parse_odometer(it.text) is not None:
                it.shot = "odometer"
            elif it.text and any(r.vin for r in vinlib.extract_vins(it.text)):
                sticker = re.search(r"MFD|GVWR|CONFORMS|TIRE", it.text.upper())
                it.shot = "door_jamb" if sticker else "vin_plate"
            if it.shot:
                it.shot_source = "rules"


def _alpr_pass(items: list[_Item], engines: Engines) -> None:
    todo = [it for it in items if it.shot is None or it.shot in ALPR_SHOTS]
    if not todo:
        return
    engines.activate("alpr")
    for it in todo:
        it.plates = engines.alpr(it.image)
        if it.shot is None:
            best = max(it.plates, key=lambda r: (r.area, r.confidence), default=None)
            frame = it.image.width * it.image.height
            it.shot = "other" if best is None else "plate" if best.area / frame >= PLATE_FILLS_FRAME else "exterior"
            it.shot_source = "rules"


def _combine(it: _Item, vision_name: str, vpic) -> PhotoFields:
    r = PhotoFields(str(it.path), notes=list(it.notes))
    a = it.answer or {}
    shot = it.shot or "other"
    conf = VISION_CONFIDENCE if it.shot_source.startswith("vision") else RULES_SHOT_CONFIDENCE
    r.fields["shot_type"] = FieldValue(shot, conf, a.get("shot_type") or "", it.shot_source or "rules")

    if it.exif.captured_at:
        r.fields["captured_at"] = FieldValue(it.exif.captured_at, EXIF_CONFIDENCE, "EXIF DateTimeOriginal", "exif")
    elif it.uploaded_at:
        r.fields["captured_at"] = FieldValue(it.uploaded_at, UPLOAD_CONFIDENCE, "upload time", "upload")
    if it.exif.gps:
        r.fields["gps"] = FieldValue(it.exif.gps, EXIF_CONFIDENCE, "EXIF GPS", "exif")
    if it.exif.device:
        r.fields["device"] = FieldValue(it.exif.device, EXIF_CONFIDENCE, "EXIF Make/Model", "exif")

    if it.barcode_vin or shot in VIN_SHOTS:
        _combine_vin(r, it, a.get("vin"), vision_name, vpic)
    if shot in ALPR_SHOTS:
        _combine_plate(r, it, a, vision_name)
    if shot == "odometer":
        ocr_value = parse_odometer(it.text) if it.text else None
        model_value = _model_odometer(a.get("odometer"))
        if ocr_value is not None:
            r.fields["odometer"] = FieldValue(ocr_value, 0.85, it.text or "", it.ocr_source)
            if model_value is not None and model_value != ocr_value:
                r.conflicts["odometer"] = {"OCR": str(ocr_value), "vision model": a.get("odometer")}
        elif model_value is not None:
            r.fields["odometer"] = FieldValue(model_value, VISION_CONFIDENCE, a.get("odometer") or "", vision_name)
    if shot == "exterior":
        color, style = normalize_color(a.get("color") or ""), normalize_body_style(a.get("body_style"))
        if color:
            r.fields["color"] = FieldValue(color, VISION_CONFIDENCE, a["color"], vision_name)
        if style:
            r.fields["body_style"] = FieldValue(style, VISION_CONFIDENCE, a["body_style"], vision_name)
    return r


def _combine_vin(r: PhotoFields, it: _Item, model_raw: str | None, vision_name: str, vpic) -> None:
    model_vin = vinlib.normalize(model_raw or "")
    model_vin = model_vin if len(model_vin) == 17 and vinlib.is_valid(model_vin) else None
    if it.barcode_vin:
        r.fields["vin"] = FieldValue(it.barcode_vin, BARCODE_CONFIDENCE, it.barcode_vin, "barcode")
        r.vin_confirmations.append(VinConfirmation("barcode", True, "barcode with a valid check digit"))
        if model_vin and model_vin != it.barcode_vin:
            r.notes.append(f"vision model read {model_vin}; the barcode wins")
        return

    results = vinlib.extract_vins(it.text) if it.text else []
    usable = [x for x in results if x.status is not vinlib.VinStatus.INVALID]
    best = max(usable, key=lambda x: x.confidence, default=None)
    S = vinlib.VinStatus
    if best is not None and best.vin:
        r.fields["vin"] = FieldValue(best.vin, best.confidence, best.raw, it.ocr_source)
        issues = r.vin_issues
        if best.status is S.UNVERIFIED:
            issues.append("VIN fails the check digit")
        elif best.status is S.REPAIRED:
            issues.append("VIN was repaired from a likely OCR misread")
        if best.vin[:3] not in WMI_MAKES:
            issues.append(f"unknown manufacturer code (WMI) {best.vin[:3]}")
        if len({x.vin for x in usable if x.vin}) > 1:
            issues.append("several VINs in the photo")
        if model_vin and model_vin != best.vin:
            issues.append(f"vision model read a different VIN ({model_vin})")
        if not issues and model_vin == best.vin:
            r.vin_confirmations.append(VinConfirmation("vision", True, "vision model read the same VIN"))
        return

    if not model_vin:
        return
    from_ocr = any(x.vin == model_vin or model_vin in x.alternatives for x in results)
    if not from_ocr:
        if vpic is False:
            return
        d = (vpic or vinlib.decode_vpic)(model_vin)
        if not (d.ok and d.clean):
            return
    r.fields["vin"] = FieldValue(model_vin, VISION_CONFIDENCE, model_raw or "", vision_name)
    r.vin_issues.append("VIN read only by the vision model")


def _combine_plate(r: PhotoFields, it: _Item, a: dict, vision_name: str) -> None:
    best = max(it.plates or [], key=lambda x: (x.area, x.confidence), default=None)
    alpr_plate = _plate_text(best.text) if best else None
    model_plate = _plate_text(a.get("plate"))
    state = (a.get("plate_state") or "").strip().upper()
    state = state if state in US_STATES else None
    if alpr_plate:
        r.fields["plate"] = FieldValue(alpr_plate, best.confidence, best.text, "alpr")
        if model_plate and model_plate != alpr_plate:
            r.conflicts["plate"] = {"ALPR": alpr_plate, "vision model": a.get("plate")}
        elif model_plate == alpr_plate:
            r.plate_confirmed = True
            if state:
                r.fields["plate_state"] = FieldValue(state, VISION_CONFIDENCE, a["plate_state"], vision_name)
    elif model_plate:
        r.fields["plate"] = FieldValue(model_plate, VISION_CONFIDENCE, a.get("plate") or "", vision_name)
        if state:
            r.fields["plate_state"] = FieldValue(state, VISION_CONFIDENCE, a["plate_state"], vision_name)
