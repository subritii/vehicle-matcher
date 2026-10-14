"""Files -> database rows, shared by tools/seed_demo.py and uploads (app/jobs.py).

Invoice and photo extraction run with every raw engine output cached (OCR, fast-alpr,
the vision model, vPIC), so known files never re-run an engine. Engines run one at a
time: invoice OCR, then (OCR stopped) the vision model, then the photo passes.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app import storage as st
from app.extract import cache as ch
from app.extract import photo as ph
from app.extract.invoice import (
    InvoiceFields,
    Page,
    _load_image,
    _rasterize_pdf,
    extract_invoice,
    fill_gaps_with_vision,
    get_ocr_engine,
    read_page,
    stop_ocr,
)
from app.match.cluster import Cluster as MatchCluster
from app.match.cluster import PhotoObs, _model_only
from app.match.score import Decision, InvoiceObs, _best
from app.models import Cluster, Invoice, MatchEvent, Photo, index_keys

SHOT_ORDER = {"vin_plate": 0, "door_jamb": 1, "plate": 2, "odometer": 3, "exterior": 4, "damage": 5,
              "interior": 6, "other": 7}
INVOICE_SIDE = 1600


class LazyOcr:
    """The shared OCR engine, started only when an image is not in the cache."""

    def __init__(self) -> None:
        self.name = os.getenv("OCR_ENGINE", "paddle").strip().lower() or "paddle"

    def __call__(self, image):
        return get_ocr_engine()(image)

    @property
    def word_level(self) -> bool:
        return bool(getattr(get_ocr_engine(), "word_level", False))

    def close(self) -> None:
        stop_ocr()


class _CacheOnlyOcr:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, image):
        raise LookupError("not in the OCR cache")


@dataclass
class Readers:
    """Cached engines for one run. ``vision`` is None for rules only."""

    cache: ch.DiskCache
    ocr_cache: Path
    vision: object | None
    vpic: object

    @classmethod
    def build(cls, settings, vision=None, vpic: bool = True) -> Readers:
        cache = ch.DiskCache(settings.extract_cache)
        return cls(cache, settings.ocr_cache, ch.CachedVision(vision, cache) if vision is not None else None,
                   ch.cached_vpic(cache) if vpic else False)

    def photo_engines(self) -> ph.Engines:
        ocr = LazyOcr()
        return ph.Engines(ocr=ch.CachedOcr(ocr, self.cache, name=ocr.name),
                          alpr=ch.CachedAlpr(ph.FastAlpr(), self.cache), vision=self.vision)

    def invoice_rules(self, path: Path) -> InvoiceFields:
        return extract_invoice(path, ocr=LazyOcr(), cache_dir=self.ocr_cache, vpic=self.vpic)

    def invoices(self, files: dict[str, Path]) -> dict[str, InvoiceFields]:
        """Rules on every file, then (OCR stopped) one vision model session for the gaps."""
        fields = {iid: self.invoice_rules(p) for iid, p in files.items()}
        if self.vision is None:
            return fields
        stop_ocr()
        with self.vision.session():
            return {iid: fill_gaps_with_vision(copy.deepcopy(f), files[iid], self.vision, vpic=self.vpic)
                    for iid, f in fields.items()}


# --- Invoice display image and field boxes ---------------------------------------------


def invoice_image(path: Path):
    """The invoice as one image (PDF pages stacked), for display."""
    from PIL import Image

    if path.suffix.lower() != ".pdf":
        return _load_image(path)
    pages = _rasterize_pdf(path)
    out = Image.new("RGB", (max(p.width for p in pages), sum(p.height for p in pages)), "white")
    y = 0
    for p in pages:
        out.paste(p, (0, y))
        y += p.height
    return out


def _alnum(text: str) -> tuple[str, list[int]]:
    keep = [(i, c) for i, c in enumerate(text.upper()) if c.isalnum()]
    return "".join(c for _, c in keep), [i for i, _ in keep]


def _variants(value) -> list[str]:
    if isinstance(value, date):
        d = value
        forms = [d.isoformat(), f"{d.month}/{d.day}/{d.year}", d.strftime("%m/%d/%Y"), d.strftime("%m/%d/%y"),
                 d.strftime("%b %d %Y"), d.strftime("%B %d %Y"), d.strftime("%d %b %Y"), d.strftime("%d %B %Y"),
                 f"{d.strftime('%b')} {d.day} {d.year}", f"{d.day} {d.strftime('%b')} {d.year}"]
    elif isinstance(value, (Decimal, float)):
        forms = [f"{value:.2f}"]
    else:
        forms = [str(value)]
    return [a for a in dict.fromkeys(_alnum(f)[0] for f in forms) if len(a) >= 2]


def _value_box(line, value) -> tuple[float, float, float, float] | None:
    text, idx = _alnum(line.text)
    for v in _variants(value):
        k = text.find(v)
        if k < 0:
            continue
        start, end = idx[k], idx[k + len(v) - 1] + 1
        segs = line.segs[line.seg_index(start): line.seg_index(end - 1) + 1]
        return (line.x_at(start), min(s.y0 for s in segs), line.x_at(end), max(s.y1 for s in segs))
    return None


def locate(page: Page, evidence: str, value) -> tuple[float, float, float, float] | None:
    """Where on the page a field was read: the value on its evidence line (or the two
    lines below, for labels with the value underneath), else the whole evidence line;
    without such a line, a distinctive value anywhere. Fractions of the page (x0, y0, x1, y1)."""
    for i, line in enumerate(page.lines):
        if line.text != evidence:
            continue
        box = None
        for nl in page.lines[i: i + 3]:
            box = _value_box(nl, value)
            if box:
                break
        if box is None:
            box = (min(s.x0 for s in line.segs), min(s.y0 for s in line.segs),
                   max(s.x1 for s in line.segs), max(s.y1 for s in line.segs))
        return _fraction(page, box)
    # Evidence that is not a line (e.g. a VIN found anywhere in the text): a distinctive
    # value (5+ characters) is looked up on every line.
    if len(_alnum(str(value))[0]) >= 5:
        for line in page.lines:
            box = _value_box(line, value)
            if box:
                return _fraction(page, box)
    return None


def _fraction(page: Page, box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (round(x0 / page.width, 4), round(y0 / page.height, 4),
            round(x1 / page.width, 4), round(y1 / page.height, 4))


def field_boxes(path: Path, f: InvoiceFields, ocr_cache: Path) -> dict[str, list[float]]:
    """Boxes for fields read from the PDF text layer or by OCR (cached pages only)."""
    pages: dict[str, Page | None] = {}

    def page_for(source: str) -> Page | None:
        if source not in pages:
            try:
                if source == "pdf_text":
                    pages[source] = read_page(path)
                else:
                    pages[source] = read_page(path, ocr=_CacheOnlyOcr(source.removeprefix("ocr:")),
                                              cache_dir=ocr_cache, force_ocr=True)
            except (LookupError, ValueError):
                pages[source] = None
        return pages[source]

    out = {}
    for name, fv in f.fields.items():
        if fv.source == "pdf_text" or fv.source.startswith("ocr:"):
            page = page_for(fv.source)
            if page is not None and page.source == fv.source and fv.evidence:
                box = locate(page, fv.evidence, fv.value)
                if box:
                    out[name] = list(box)
    return out


# --- Observations <-> JSON -------------------------------------------------------------


def invoice_obs_to_json(o: InvoiceObs) -> dict:
    d = asdict(o)
    d["date"] = o.date.isoformat() if o.date else None
    return d


def invoice_obs_from_json(d: dict) -> InvoiceObs:
    d = dict(d)
    d["date"] = date.fromisoformat(d["date"]) if d.get("date") else None
    return InvoiceObs(**d)


def photo_obs_to_json(o: PhotoObs) -> dict:
    d = asdict(o)
    d["time"] = o.time.isoformat() if o.time else None
    d["gps"] = list(o.gps) if o.gps else None
    return d


def photo_obs_from_json(d: dict) -> PhotoObs:
    d = dict(d)
    d["time"] = datetime.fromisoformat(d["time"]) if d.get("time") else None
    d["gps"] = tuple(d["gps"]) if d.get("gps") else None
    return PhotoObs(**d)


# --- Saving -----------------------------------------------------------------------------


def add_invoice(db, storage: st.LocalStorage, iid: str, path: Path, kind: str, f: InvoiceFields,
                ocr_cache: Path, prefix: str) -> Invoice:
    orig = storage.put_file(f"{prefix}/invoices/{iid}{path.suffix.lower()}", path)
    view = storage.put_image(f"{prefix}/invoices/{iid}_view.jpg", invoice_image(path), INVOICE_SIDE)
    data = f.to_dict()
    data["boxes"] = field_boxes(path, f, ocr_cache)
    total, when = f.value("total"), f.value("invoice_date")
    mileage = f.value("mileage")
    inv = Invoice(id=iid, vendor=f.value("vendor"), invoice_no=f.value("invoice_no"),
                  date=when if isinstance(when, date) else None,
                  mileage=mileage if isinstance(mileage, int) else None,
                  total=f"{total:.2f}" if isinstance(total, (Decimal, float)) else (str(total) if total else None),
                  file_path=orig, image_path=view, file_kind=kind, extracted_json=data,
                  obs_json=invoice_obs_to_json(InvoiceObs.from_invoice(iid, f)))
    index_keys(inv)
    db.add(inv)
    return inv


@dataclass
class PhotoIn:
    path: Path
    uploaded_at: datetime
    uploader: str | None


def add_photos(db, storage: st.LocalStorage, prefix: str, items: list[tuple[PhotoIn, ph.PhotoFields]]
               ) -> list[tuple[Photo, PhotoObs]]:
    out = []
    for spec, pf in items:
        im = _load_image(spec.path)
        name = st.safe_name(spec.path.stem)
        shot = pf.value("shot_type")
        gps = pf.value("gps")
        row = Photo(file_path=storage.put_image(f"{prefix}/photos/{name}.jpg", im, st.DISPLAY_SIDE),
                    thumb_path=storage.put_image(f"{prefix}/thumbs/{name}.jpg", im, st.THUMB_SIDE),
                    shot_type=shot, captured_at=pf.value("captured_at"),
                    time_from_exif="captured_at" in pf.fields and pf.fields["captured_at"].source == "exif",
                    gps=list(gps) if gps else None, uploader=spec.uploader, read_vin=pf.value("vin"),
                    read_plate=pf.value("plate"), read_odometer=pf.value("odometer"),
                    extracted_json=pf.to_dict(), sort_key=SHOT_ORDER.get(shot, 9))
        db.add(row)
        db.flush()
        obs = PhotoObs.from_photo(str(row.id), pf, uploader=spec.uploader)
        row.obs_json = photo_obs_to_json(obs)
        out.append((row, obs))
    return out


def cluster_summary(c: MatchCluster, vpic) -> dict:
    """What the photos say about the vehicle, for the review screen."""
    vin_p = next(iter(_best([p for p in c.photos if p.vin], "vin_trusted", "vin_source")), None)
    plate_p = next(iter(_best([p for p in c.photos if p.plate], "plate_trusted", "plate_source")), None)
    out = {"vin": vin_p.vin if vin_p else None, "vin_trusted": bool(vin_p and vin_p.vin_trusted),
           "vin_source": vin_p.vin_source if vin_p else "",
           "plate": plate_p.plate if plate_p else None, "plate_state": plate_p.plate_state if plate_p else None,
           "plate_trusted": bool(plate_p and plate_p.plate_trusted),
           "color": c.color, "body_style": c.body_style, "odometer": c.odometer,
           "time": c.time.isoformat() if c.time else None, "time_from_exif": c.time_from_exif,
           "uploaders": sorted(c.uploaders), "make": None, "model": None, "year": None}
    if vpic and vin_p and not _model_only(vin_p.vin_source):
        d = vpic(vin_p.vin)
        if d.ok:
            out.update(make=d.make, model=d.model, year=d.year)
    return out


def apply_match(db, clusters: list[MatchCluster], decisions: list[Decision], vpic, id_prefix: str = "",
                truth: dict[str, str | None] | None = None) -> list[Cluster]:
    """Store clusters and their decisions; every decision writes a match event."""
    from app.services import link_cluster

    rows = []
    for c, d in zip(clusters, decisions):
        cid = f"{id_prefix}{c.id}"
        reasons = list(d.reasons)
        if c.status == "unassigned":
            reasons = [f"This photo {c.reason}"]
        row = Cluster(id=cid, status="review" if d.status == "linked" else d.status, score=d.score,
                      candidates_json=[x.to_dict() for x in d.candidates], reasons_json=reasons,
                      summary_json=cluster_summary(c, vpic), truth_invoice_id=(truth or {}).get(c.id))
        db.add(row)
        db.flush()
        for p in c.photos:
            db.get(Photo, int(p.id)).cluster_id = cid
        if d.status == "linked":
            cand = d.candidates[0].to_dict() if d.candidates else {}
            link_cluster(db, row, db.get(Invoice, d.invoice_id), decided_by="auto", action="linked", how=d.decided_by,
                         score=d.score, signals=cand.get("signals", []))
        else:
            top = d.candidates[0] if d.candidates else None
            db.add(MatchEvent(cluster_id=cid, invoice_id=top.invoice_id if top else None, action=d.status,
                              score=d.score, signals_json=[vars(s) for s in top.signals] if top else [],
                              decided_by="auto"))
        rows.append(row)
    return rows
