"""Find invoice from photo: score photos of one vehicle against every invoice.

A lookup is read-only: the photos stay in a temporary folder (data/lookups/<token>/) and
the result in memory. Nothing is written to the database unless the user saves the
photos to an invoice. Extraction runs on the app's single worker, so OCR and the vision
model never run together.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from app.match import config as cfg
from app.match.cluster import Cluster as MatchCluster
from app.match.cluster import PhotoObs, fold_plate
from app.match.score import Candidate, InvoiceObs, score_pair
from app.storage import safe_name

log = logging.getLogger(__name__)

ACCEPTED = {".jpg": "JPG", ".jpeg": "JPG", ".png": "PNG", ".heic": "HEIC", ".heif": "HEIC"}
MAX_PHOTOS = 20
MAX_AGE = timedelta(hours=1)

STEPS = (
    ("read", "Opening the photos…"),
    ("photo_ai", "Looking at the photos…"),
    ("photo_ocr", "Reading the VIN and mileage…"),
    ("photo_alpr", "Reading the license plate…"),
    ("compare", "Finding the matching invoice…"),
)
PHOTO_STEPS = {"vision": "photo_ai", "ocr": "photo_ocr", "alpr": "photo_alpr"}


@dataclass
class Lookup:
    token: str
    dir: Path
    names: list[str]
    created: datetime = field(default_factory=datetime.now)
    status: str = "queued"  # queued | running | done | failed
    steps: list[dict] = field(default_factory=lambda: [{"key": k, "label": label, "status": "pending", "detail": ""}
                                                       for k, label in STEPS])
    error: str | None = None
    fields: list = field(default_factory=list)  # PhotoFields, kept for "Save to this invoice"
    vpic: object = None
    result: dict | None = None

    @property
    def files(self) -> list[Path]:
        return [self.dir / f"{i:02d}_{safe_name(Path(n).stem)}{Path(n).suffix.lower()}" for i, n in enumerate(self.names)]

    def thumb(self, i: int) -> Path:
        return self.dir / f"thumb_{i:02d}.jpg"

    def set_step(self, key: str, status: str, detail: str | None = None) -> None:
        for s in self.steps:
            if s["key"] == key:
                s["status"] = status
                if detail is not None:
                    s["detail"] = detail

    @property
    def current(self) -> dict | None:
        return next((s for s in self.steps if s["status"] == "running"), None)

    @property
    def finished(self) -> int:
        return sum(s["status"] in ("done", "skipped") for s in self.steps)


class LookupStore:
    """Lookups in memory, their files on disk; both dropped after an hour or a save."""

    def __init__(self) -> None:
        self._items: dict[str, Lookup] = {}
        self._lock = threading.Lock()

    def create(self, root: Path, names: list[str]) -> Lookup:
        self.cleanup()
        token = secrets.token_urlsafe(12)
        item = Lookup(token, Path(root) / token, list(names))
        item.dir.mkdir(parents=True)
        with self._lock:
            self._items[token] = item
        return item

    def get(self, token: str) -> Lookup | None:
        with self._lock:
            return self._items.get(token)

    def drop(self, token: str) -> None:
        with self._lock:
            item = self._items.pop(token, None)
        if item is not None:
            shutil.rmtree(item.dir, ignore_errors=True)

    def cleanup(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        with self._lock:
            old = [t for t, x in self._items.items() if now - x.created > MAX_AGE]
        for t in old:
            self.drop(t)


# --- Scoring and plain-language reasons ---------------------------------------------------


def _days(n: int) -> str:
    return f"{n} day{'s' if n != 1 else ''}"


def explain(cand: Candidate, c: MatchCluster, inv: InvoiceObs) -> tuple[list[str], list[str]]:
    """Why an invoice fits (reasons) and what speaks against it (warnings), in plain words."""
    sig = {s.name: s for s in cand.signals if s.points > 0}
    reasons: list[str] = []
    if "vin" in sig:
        d = sig["vin"].detail
        reasons.append("VIN matches" if cand.hard == "vin" else
                       "VIN matches except one easily confused character" if "confusable" in d else
                       "VIN matches the vision model's reading" if "vision model" in d else
                       "VIN matches (read by one reader only)")
    if "plate" in sig:
        d = sig["plate"].detail
        reasons.append("plate matches" if "trusted on both" in d else
                       "plate matches the vision model's reading" if "vision model" in d else
                       "plate matches (read by one reader only)")
    if "make_model" in sig and "color" in sig:
        reasons.append("color and model match")
    elif "make_model" in sig:
        reasons.append("make and model match")
    elif "color" in sig:
        reasons.append("color matches")
    if "odometer" in sig and c.odometer is not None and inv.mileage is not None:
        diff = abs(c.odometer - inv.mileage)
        reasons.append("odometer matches the invoice" if diff == 0 else f"odometer within {diff:,} mi of the invoice")
    if "date" in sig and c.time and inv.date:
        d = (c.time.date() - inv.date).days
        when = ("the same day as" if d == 0 else f"{_days(d)} after" if d > 0 else f"{_days(-d)} before")
        reasons.append(f"photo taken {when} the invoice" + ("" if c.time_from_exif else " (upload time)"))

    warnings: list[str] = []
    if cand.misread:  # "invoice VIN likely misread: X vs photo Y (3 differences)"
        what = "VIN" if cand.misread.startswith("invoice VIN") else "plate"
        warnings.append(f"invoice {what} looks misread ({cand.misread.rsplit('(', 1)[-1]}")
    for x in cand.conflicts:
        if x.startswith("color"):
            warnings.append(f"color differs (photo {c.color}, invoice {inv.color})")
        elif x.startswith("odometer"):
            warnings.append(f"odometer differs ({c.odometer:,} vs {inv.mileage:,} mi)")
        elif x.startswith("make/model"):
            warnings.append(f"make/model differs (invoice says {inv.make} {inv.model})")
        elif x.startswith("VIN") and not cand.misread:
            warnings.append("VIN differs")
        elif x.startswith("plate") and not cand.misread:
            warnings.append("plate differs" if "state" not in x else "plate state differs")
    return reasons, warnings


def confidence(score: float) -> str:
    return "High" if score >= cfg.AUTO_SCORE else "Medium" if score >= cfg.REVIEW_MIN else "Low"


# --- Search: indexed candidates, then the matcher's own scoring on the short list ---------


@dataclass
class Found:
    rows: list  # app.models.Invoice
    searches: list[str]  # what was looked up, in plain words


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{'no' if n == 0 else n} {one if n == 1 else (many or one + 'es' if one.endswith('ch') else many or one + 's')}"


_PLATE_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # folded: O->0, I->1, Q->0


def _plate_keys(plate: str) -> set[str]:
    """The plate, every cut-short prefix the misread rule accepts, and every plate one edit away."""
    keys = {plate} | {plate[:k] for k in range(cfg.MISREAD_PLATE_MIN_PREFIX, len(plate))}
    for i in range(len(plate) + 1):
        if i < len(plate):
            keys.add(plate[:i] + plate[i + 1:])  # one character dropped
            keys |= {plate[:i] + ch + plate[i + 1:] for ch in _PLATE_CHARS}  # one character changed
        keys |= {plate[:i] + ch + plate[i:] for ch in _PLATE_CHARS}  # one character added
    return keys


def candidate_rows(db, c: MatchCluster, vpic=None) -> Found:
    """The invoices worth scoring, found through indexes. Every invoice that could pass the
    lookup threshold (cfg.LOOKUP_MIN), match on a VIN or plate, or count as a likely misread
    is included: VINs and plates (exact, a few characters off, cut short), odometer within
    the scoring range, and color or make within a week of the photo date."""
    from sqlalchemy import and_, func, or_, select

    from app.match.score import _edits
    from app.models import Invoice, vin_pieces

    total = db.scalar(select(func.count()).select_from(Invoice)) or 0
    found: dict[str, object] = {}
    searches: list[str] = []

    def add(rows) -> None:
        for r in rows:
            found.setdefault(r.id, r)

    def load(ids) -> list:
        ids = list(ids)
        return db.scalars(select(Invoice).where(Invoice.id.in_(ids))).all() if ids else []

    for vin in sorted({p.vin.upper() for p in c.photos if p.vin}):
        exact = db.scalars(select(Invoice).where(Invoice.vin == vin)).all()
        pieces = vin_pieces(vin)
        near_ids = [i for i, v in db.execute(select(Invoice.id, Invoice.vin).where(
            or_(Invoice.vin_p1 == pieces[0], Invoice.vin_p2 == pieces[1], Invoice.vin_p3 == pieces[2],
                Invoice.vin_p4 == pieces[3]), Invoice.vin != vin)) if _edits(v, vin) <= cfg.MISREAD_VIN_EDITS]
        add(exact)
        add(load(near_ids))
        searches.append(f"Looked up VIN {vin} among {total} invoices → {_count(len(exact), 'exact match')}"
                        + (f", {len(near_ids)} with a VIN a few characters off" if near_ids else ""))

    for plate in sorted({fold_plate(p.plate) for p in c.photos if p.plate}):
        rows = db.scalars(select(Invoice).where(Invoice.plate_key.in_(_plate_keys(plate)))).all()
        exact = [r for r in rows if r.plate_key == plate]
        add(rows)
        searches.append(f"Looked up plate {plate} among {total} invoices → {_count(len(exact), 'exact match')}"
                        + (f", {len(rows) - len(exact)} nearly the same" if len(rows) > len(exact) else ""))

    if c.odometer is not None:
        rows = db.scalars(select(Invoice).where(
            Invoice.mileage.between(c.odometer - cfg.ODOMETER_MILES, c.odometer + cfg.ODOMETER_MILES))).all()
        add(rows)
        searches.append(f"Odometer {c.odometer:,} miles → {_count(len(rows), 'invoice')} within "
                        f"{cfg.ODOMETER_MILES} miles")

    make = None
    if vpic:
        vins = [p.vin for p in c.photos if p.vin and not p.vin_source.startswith("vision")]
        d = vpic(vins[0]) if vins else None
        make = d.make.lower() if d is not None and d.ok and d.make else None
    if c.time is not None and (c.color or make):
        lo, hi = c.time.date() - timedelta(days=cfg.DATE_ZERO_DAYS), c.time.date() + timedelta(days=cfg.DATE_ZERO_DAYS)
        same = [Invoice.color == c.color] if c.color else []
        same += [Invoice.make_key == make] if make else []
        near_date = db.scalars(select(Invoice).where(Invoice.date.between(lo, hi), or_(*same))).all()
        add(near_date)
        what = " or ".join(x for x in (c.color, make and make.title()) if x)
        searches.append(f"{what[:1].upper() + what[1:]}, dated within a week of the photos → "
                        f"{_count(len(near_date), 'invoice')}")
        if c.color and make:
            add(db.scalars(select(Invoice).where(and_(Invoice.make_key == make, Invoice.color == c.color))).all())
    if not searches:
        searches.append("Nothing in the photos to search by (no VIN, plate, odometer or color was read)")
    searches.append(f"Compared {_count(len(found), 'invoice')} closely, detail by detail" if found
                    else "No invoice was close enough to compare in detail")
    return Found(list(found.values()), searches)


def _item(cand: Candidate, c: MatchCluster, inv: InvoiceObs, row) -> dict:
    agree, disagree = explain(cand, c, inv)
    return {"invoice_id": row.id, "score": cand.score, "confidence": confidence(cand.score),
            "pct": min(100, round(cand.score)), "agree": agree, "disagree": disagree,
            "highlight": cand.hard, "linked": row.status == "linked",
            "id_evidence": bool(cand.misread or any(sig.name in ("vin", "plate") and sig.points > 0
                                                    for sig in cand.signals))}


def rank(c: MatchCluster, rows, vpic=None) -> dict:
    """The matcher's scores for these invoices: one confident match (a VIN or plate confirmed
    on both sides, on exactly one invoice), else the best candidates, else none."""
    from app.pipeline import invoice_obs_from_json

    scored = []
    for row in rows:
        inv = invoice_obs_from_json(row.obs_json)
        scored.append((score_pair(c, inv, vpic), inv, row))
    hard = [x for x in scored if x[0].hard and not x[0].veto and not x[0].id_conflict]
    if len(hard) == 1:
        cand, inv, row = hard[0]
        out = _item(cand, c, inv, row)
        out["confidence"], out["pct"] = "Confident", 100
        return {"kind": "confident", "items": [out]}
    fits = [x for x in scored if not x[0].veto and (x[0].score >= cfg.LOOKUP_MIN or x[0].misread)]
    fits.sort(key=lambda x: (not x[0].hard, not x[0].misread, -x[0].score, x[2].id))
    if not fits:
        return {"kind": "none", "items": []}
    return {"kind": "possible", "items": [_item(cand, c, inv, row) for cand, inv, row in fits[: cfg.TOP_CANDIDATES]]}


def _describe(db, c: MatchCluster, read: dict) -> dict | None:
    """Invoices that fit what an exterior or odometer photo shows (no VIN or plate)."""
    from sqlalchemy import func, select

    from app.models import Invoice

    where, words = [], []
    if c.color:
        where.append(Invoice.color == c.color)
        words.append(c.color)
    if read.get("make"):
        where.append(Invoice.make_key == read["make"].lower())
        words.append(read["make"].title() + (f" {read['model']}" if read.get("model") else ""))
    if c.odometer is not None:
        where.append(Invoice.mileage.between(c.odometer - cfg.ODOMETER_MILES, c.odometer + cfg.ODOMETER_MILES))
    if not where:
        return None
    count = db.scalar(select(func.count()).select_from(Invoice).where(*where)) or 0
    what = " ".join(words) + (" car" if words else "car")
    if c.odometer is not None:
        what += f" with about {c.odometer:,} miles"
    rows = db.scalars(select(Invoice).where(*where).order_by(Invoice.date.desc()).limit(200)).all()
    return {"count": count, "what": what.strip(), "rows": rows}


def find_invoices(obs: list[PhotoObs], db, vpic=None) -> dict:
    """Photos of one vehicle -> {"kind": "confident" | "possible" | "many" | "none", "items",
    "read", "searches", "why", "fits"}. "many": no VIN or plate to go on and several invoices
    fit the description, so none is picked."""
    from app.pipeline import cluster_summary

    c = MatchCluster("lookup", obs)
    read = cluster_summary(c, vpic)
    if not read["plate_trusted"] and not any(p.plate and p.shot_type == "plate" for p in obs):
        read["plate"] = read["plate_state"] = None  # "plates" on exteriors are often badges or frames
    found = candidate_rows(db, c, vpic)
    res = rank(c, found.rows, vpic)
    res.update(read=read, searches=found.searches, fits=None)
    identified = res["kind"] == "confident" or any(x["id_evidence"] for x in res["items"])
    if not identified:
        fits = _describe(db, c, read)
        if fits and fits["count"] >= 2:  # a few of them, best first, whatever their score
            scored = sorted(((score_pair(c, _obs(r), vpic), r) for r in fits["rows"]), key=lambda x: (-x[0].score, x[1].id))
            top = [_item(cand, c, _obs(r), r) for cand, r in scored[:5]]
            res.update(kind="many", items=top, fits={"count": fits["count"], "what": fits["what"]})
        elif fits and fits["count"] == 1 and res["kind"] == "none":  # the only invoice that fits
            row = fits["rows"][0]
            res.update(kind="possible", items=[_item(score_pair(c, _obs(row), vpic), c, _obs(row), row)])
    res["why"] = _why(res, read, obs)
    return res


def _obs(row) -> InvoiceObs:
    from app.pipeline import invoice_obs_from_json

    return invoice_obs_from_json(row.obs_json)


def _why(res: dict, read: dict, obs: list[PhotoObs]) -> str:
    kind = res["kind"]
    if kind == "confident":
        what = "VIN" if res["items"][0]["highlight"] == "vin" else "license plate and state"
        return (f"The {what} in the photos is confirmed (two independent readers agree) and exactly one "
                f"invoice has the same confirmed {what}.")
    if kind == "many":
        f = res["fits"]
        return (f"{f['count']} invoices fit this description ({f['what']}), and nothing in the photos tells them "
                "apart, so the tool doesn't pick one. Add a photo of the VIN sticker or the license plate to find "
                "the exact invoice.")
    if kind == "possible":
        top = res["items"][0]
        if any("looks misread" in w for w in top["disagree"]):
            return ("The VIN or plate on this invoice is a few characters off from the photos, "
                    "probably misread from the invoice. Check before saving.")
        return ("No confirmed VIN or plate matched exactly one invoice, but these invoices share "
                "details with the photos. Check before saving.")
    if not any(p.vin or p.plate or p.color or p.odometer is not None for p in obs):
        return "Nothing that identifies the car could be read from these photos."
    return "No invoice shares enough details with the photos. The invoice may not be in the system yet."


# --- What was read from each photo ----------------------------------------------------------

_READER = {"barcode": "barcode", "alpr": "plate reader", "exif": "photo info", "rules": "rules", "upload": "upload"}
_SHOT = {"vin_plate": "VIN plate", "door_jamb": "Door sticker", "plate": "License plate", "odometer": "Odometer",
         "exterior": "Outside of the car", "interior": "Interior", "damage": "Damage", "other": "Other photo"}


def reader_name(source: str) -> str:
    if source.startswith("ocr:"):
        return "OCR"
    if source.startswith("vision"):
        return "AI"
    return _READER.get(source, source or "-")


def photo_reads(fields: list, names: list[str]) -> list[dict]:
    """Per photo: its type, and each value read with the reader that produced it."""
    out = []
    for pf, name in zip(fields, names):
        f = pf.fields
        shot = f.get("shot_type")
        reads = []
        for key, label, fmt in (("vin", "VIN", str), ("plate", "Plate", str), ("plate_state", "State", str),
                                ("odometer", "Odometer", lambda v: f"{v:,} miles"),
                                ("color", "Color", lambda v: str(v).capitalize()), ("body_style", "Body style", str)):
            if key in f:
                reads.append((label, fmt(f[key].value), reader_name(f[key].source)))
        out.append({"name": name, "type": _SHOT.get(shot.value if shot else "other", "Other photo"),
                    "type_by": reader_name(shot.source) if shot else "-", "reads": reads,
                    "notes": list(pf.notes)})
    return out


# --- Running a lookup and saving it --------------------------------------------------------


def default_reader(settings, vision_factory):
    """Photo extraction with the cached engines: read_photos(paths, progress, uploaded_at)."""

    def read_photos(paths, progress, uploaded_at):
        from app.extract.photo import extract_photos
        from app.pipeline import Readers

        readers = Readers.build(settings, vision=vision_factory())
        fields = extract_photos(paths, readers.photo_engines(), uploaded_at=uploaded_at, vpic=readers.vpic,
                                progress=progress)
        return fields, readers.vpic

    return read_photos


def run_lookup(item: Lookup, read_photos, sessions, vision_on: bool) -> None:
    """Runs on the app's single worker thread: one engine at a time, never next to an upload."""
    from app.extract.invoice import _load_image

    try:
        item.status = "running"
        item.set_step("read", "running")
        for i, (path, name) in enumerate(zip(item.files, item.names)):
            try:
                im = _load_image(path)
            except Exception:
                raise ValueError(f"could not open {name} as an image") from None
            im.thumbnail((480, 480))
            im.save(item.thumb(i), "JPEG", quality=85)
        n = len(item.files)
        item.set_step("read", "done", f"{n} photo{'s' * (n != 1)}")

        def progress(step: str) -> None:
            if (cur := item.current) is not None:
                item.set_step(cur["key"], "done")
            key = PHOTO_STEPS[step]
            if step == "vision" and not vision_on:
                item.set_step(key, "skipped", "AI is turned off")
            else:
                item.set_step(key, "running")

        fields, vpic = read_photos(item.files, progress, item.created)
        if (cur := item.current) is not None:
            item.set_step(cur["key"], "done")
        item.set_step("compare", "running")
        obs = [PhotoObs.from_photo(str(i), pf) for i, pf in enumerate(fields)]
        with sessions() as db:
            result = find_invoices(obs, db, vpic)
        result["photos"] = photo_reads(fields, item.names)
        item.fields, item.vpic, item.result = fields, vpic, result
        found = {"confident": "Found it", "possible": f"{len(result['items'])} possible",
                 "many": f"{(result['fits'] or {}).get('count', 0)} fit", "none": "No match"}[result["kind"]]
        item.set_step("compare", "done", found)
        item.status = "done"
    except BaseException as e:  # the page must never wait forever, whatever happened
        log.exception("photo lookup %s failed", item.token)
        try:
            why = str(e)
        except Exception:
            why = ""
        item.status, item.error = "failed", (why or type(e).__name__)[:300]
        if (cur := item.current) is not None:
            item.set_step(cur["key"], "failed")
        if not isinstance(e, Exception):
            raise


def save(db, storage, item: Lookup, invoice_id: str) -> None:
    """Store the lookup's photos with ``invoice_id``: added to its photo set if it has one,
    else linked as a new set. Writes a match event either way."""
    from sqlalchemy import select

    from app.models import Cluster, Invoice, MatchEvent
    from app.pipeline import PhotoIn, add_photos, cluster_summary, invoice_obs_from_json
    from app.services import ActionError, link_cluster

    inv = db.get(Invoice, invoice_id)
    if inv is None:
        raise ActionError("invoice not found")
    rows = add_photos(db, storage, f"found/{item.token[:10]}",
                      [(PhotoIn(p, item.created, "Photo lookup"), pf) for p, pf in zip(item.files, item.fields)])
    existing = db.scalar(select(Cluster).where(Cluster.invoice_id == invoice_id))
    if existing is not None:
        for row, _ in rows:
            row.cluster_id = existing.id
        db.add(MatchEvent(cluster_id=existing.id, invoice_id=invoice_id, action="add_photos", decided_by="user"))
        return
    mc = MatchCluster(f"f-{item.token[:8].lower()}", [obs for _, obs in rows])
    cand = score_pair(mc, invoice_obs_from_json(inv.obs_json), item.vpic or None)
    cluster = Cluster(id=mc.id, status="review", score=cand.score, candidates_json=[cand.to_dict()],
                      reasons_json=["saved from Find invoice from photo"], summary_json=cluster_summary(mc, item.vpic))
    db.add(cluster)
    db.flush()
    for row, _ in rows:
        row.cluster_id = cluster.id
    link_cluster(db, cluster, inv, decided_by="user", action="lookup", score=cand.score,
                 signals=[vars(s) for s in cand.signals])
