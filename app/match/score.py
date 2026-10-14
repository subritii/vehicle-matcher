"""Score photo clusters against invoices and decide: linked, review or unmatched.

Weights and thresholds live in app/match/config.py. Hard identifiers (auto-link on
their own) are only VINs and plates trusted on both sides; a value read by one reader
only scores points, and one read only by the vision model is a weak signal. A trusted
VIN or plate that disagrees is a veto. Wrong auto-links are worse than extra reviews:
any disagreement on an identifier blocks the auto-link, and a score-based link also
needs no disagreement at all. An invoice VIN or plate that is close to a trusted photo
read (likely misread) puts that invoice on top of a review, never an auto-link.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.match import config as cfg
from app.match.cluster import Cluster, _model_only, fold_plate

VIN_HARD_POINTS = 100  # display only: a hard VIN match links on its own


@dataclass
class InvoiceObs:
    """What matching needs from one invoice (app.extract.invoice.InvoiceFields)."""

    id: str
    vin: str | None = None
    vin_trusted: bool = False
    vin_source: str = ""
    plate: str | None = None
    plate_state: str | None = None
    plate_trusted: bool = False  # False: read only by the vision model
    make: str | None = None
    model: str | None = None
    year: int | None = None
    color: str | None = None
    mileage: int | None = None
    date: date | None = None

    @classmethod
    def from_invoice(cls, iid: str, f) -> InvoiceObs:
        v = {name: fv.value for name, fv in f.fields.items()}
        return cls(iid, vin=v.get("vin"), vin_trusted=f.vin_trusted,
                   vin_source=f.fields["vin"].source if "vin" in f.fields else "",
                   plate=v.get("plate"), plate_state=v.get("plate_state"), plate_trusted=f.plate_trusted,
                   make=v.get("make"), model=v.get("model"), year=v.get("year"), color=v.get("color"),
                   mileage=v.get("mileage"), date=v.get("invoice_date"))


@dataclass
class Signal:
    name: str
    points: float
    detail: str


@dataclass
class Candidate:
    invoice_id: str
    signals: list[Signal] = field(default_factory=list)
    hard: str | None = None  # "vin" or "plate": trusted on both sides and equal
    veto: str | None = None
    conflicts: list[str] = field(default_factory=list)
    id_conflict: bool = False  # a conflict on VIN, plate or plate state
    misread: str | None = None  # invoice VIN/plate likely misread: review, never auto-link

    @property
    def score(self) -> float:
        return round(sum(s.points for s in self.signals), 1)

    def add(self, name: str, points: float, detail: str) -> None:
        self.signals.append(Signal(name, points, detail))

    def conflict(self, text: str, identifier: bool = False) -> None:
        self.conflicts.append(text)
        self.id_conflict |= identifier

    def to_dict(self) -> dict:
        return {"invoice_id": self.invoice_id, "score": self.score, "hard": self.hard, "veto": self.veto,
                "misread": self.misread, "signals": [vars(s) for s in self.signals], "conflicts": self.conflicts}


# --- Signals -------------------------------------------------------------------------


def _one_confusable_apart(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diff) == 1 and frozenset(diff[0]) in cfg.VIN_CONFUSABLE_PAIRS


def _best(reads: list, trusted_attr: str, source_attr: str):
    """Trusted reads first, then single-reader reads, then model-only ones."""
    return sorted(reads, key=lambda p: (not getattr(p, trusted_attr), _model_only(getattr(p, source_attr))))


def _vin(c: Cluster, inv: InvoiceObs, cand: Candidate) -> None:
    reads = _best([p for p in c.photos if p.vin], "vin_trusted", "vin_source")
    if not reads or not inv.vin:
        return
    inv_weak = _model_only(inv.vin_source)
    for p in reads:  # the first read that matches decides the points
        weak = inv_weak or _model_only(p.vin_source)
        if p.vin == inv.vin:
            if p.vin_trusted and inv.vin_trusted:
                cand.hard = "vin"
                cand.add("vin", VIN_HARD_POINTS, f"{p.vin}, trusted on both sides")
            elif weak:
                cand.add("vin", cfg.MODEL_ONLY_ID, f"{p.vin}, read only by the vision model")
            else:
                cand.add("vin", cfg.VIN_ONE_READER, f"{p.vin}, read by one reader only")
            return
        if not weak and _one_confusable_apart(p.vin, inv.vin):
            cand.add("vin", cfg.VIN_CONFUSABLE, f"{p.vin} vs invoice {inv.vin}: one confusable character")
            return
    p = reads[0]
    if _model_only(p.vin_source) or inv_weak:
        return
    if p.vin_trusted and inv.vin_trusted:
        cand.veto = f"VIN {p.vin} vs invoice {inv.vin}"
    cand.conflict(f"VIN {p.vin} vs invoice {inv.vin}", identifier=True)


def _plate(c: Cluster, inv: InvoiceObs, cand: Candidate) -> None:
    reads = _best([p for p in c.photos if p.plate], "plate_trusted", "plate_source")
    if not reads or not inv.plate:
        return
    want = fold_plate(inv.plate)
    for p in reads:
        if fold_plate(p.plate) != want:
            continue
        if not inv.plate_trusted or _model_only(p.plate_source):
            cand.add("plate", cfg.MODEL_ONLY_ID, f"{p.plate}, read only by the vision model")
            return
        if p.plate_state and inv.plate_state and p.plate_state != inv.plate_state:
            cand.conflict(f"plate {p.plate}: state {p.plate_state} vs invoice {inv.plate_state}", identifier=True)
            return
        if p.plate_trusted:
            both_states = bool(p.plate_state and inv.plate_state)
            if both_states:
                cand.hard = cand.hard or "plate"
            cand.add("plate", cfg.PLATE_STATE_EXACT,
                     f"{p.plate} {p.plate_state or '(state unknown)'}, trusted on both sides")
        else:
            cand.add("plate", cfg.PLATE_ONE_READER, f"{p.plate} vs invoice {inv.plate}: read by one reader only")
        return
    # No read matches. Only a close-up read by fast-alpr disagrees in earnest.
    p = reads[0]
    if p.plate_source != "alpr" or p.shot_type != "plate" or not inv.plate_trusted:
        return
    if p.plate_trusted:
        cand.veto = cand.veto or f"plate {p.plate} vs invoice {inv.plate}"
    cand.conflict(f"plate {p.plate} vs invoice {inv.plate}", identifier=True)


def _make_model(c: Cluster, inv: InvoiceObs, cand: Candidate, vpic) -> None:
    if vpic is None or not inv.make or not inv.model:
        return
    vins = [p.vin for p in _best([p for p in c.photos if p.vin], "vin_trusted", "vin_source")
            if not _model_only(p.vin_source)]
    if not vins:
        return
    d = vpic(vins[0])
    if not d.ok or not d.make or not d.model:
        return
    same = d.make.lower() == inv.make.lower() and (
        inv.model.lower() in d.model.lower() or d.model.lower() in inv.model.lower())
    if same:
        cand.add("make_model", cfg.MAKE_MODEL, f"{d.make} {d.model} (vPIC)")
    else:
        cand.conflict(f"make/model {d.make} {d.model} (vPIC) vs invoice {inv.make} {inv.model}")


def _date(c: Cluster, inv: InvoiceObs, cand: Candidate) -> None:
    if c.time is None or inv.date is None:
        return
    days = abs((inv.date - c.time.date()).days)
    span = cfg.DATE_ZERO_DAYS - cfg.DATE_FULL_DAYS
    pts = cfg.DATE_MAX * max(0.0, min(1.0, 1 - (days - cfg.DATE_FULL_DAYS) / span))
    how = "EXIF"
    if not c.time_from_exif:
        pts *= cfg.UPLOAD_TIME_FACTOR
        how = "upload time"
    if pts:
        cand.add("date", round(pts, 1), f"photos {days} day(s) from invoice date ({how})")


def _odometer(c: Cluster, inv: InvoiceObs, cand: Candidate) -> None:
    if c.odometer is None or inv.mileage is None:
        return
    diff = abs(c.odometer - inv.mileage)
    if diff <= cfg.ODOMETER_MILES:
        pts = cfg.ODOMETER_MODEL_ONLY if c.odometer_model_only else cfg.ODOMETER
        cand.add("odometer", pts, f"{c.odometer} vs invoice {inv.mileage}")
    else:
        cand.conflict(f"odometer {c.odometer} vs invoice {inv.mileage}")


def _color(c: Cluster, inv: InvoiceObs, cand: Candidate) -> None:
    if c.color and inv.color:
        if c.color == inv.color:
            cand.add("color", cfg.COLOR, c.color)
        else:
            cand.conflict(f"color {c.color} vs invoice {inv.color}")


def _edits(a: str, b: str) -> int:
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _misread(c: Cluster, inv: InvoiceObs, cand: Candidate) -> None:
    """A trusted photo VIN/plate close to the invoice's: the invoice was likely misread.

    Only with make/model or color agreeing. The candidate goes to review on top, never
    auto-links; a near plate replaces the plate veto (a cut-short plate is not a conflict).
    """
    if not any(s.name in ("make_model", "color") for s in cand.signals):
        return
    if cand.veto and not cand.veto.startswith("plate"):
        return
    if inv.vin and not inv.vin_trusted and not any(s.name == "vin" for s in cand.signals):
        for p in c.photos:
            if p.vin_trusted and p.vin and p.vin != inv.vin:
                n = _edits(p.vin, inv.vin)
                if n <= cfg.MISREAD_VIN_EDITS:
                    cand.misread = f"invoice VIN likely misread: {inv.vin} vs photo {p.vin} ({n} differences)"
                    return
    if inv.plate and not any(s.name == "plate" for s in cand.signals):
        want = fold_plate(inv.plate)
        for p in c.photos:
            if not (p.plate_trusted and p.plate):
                continue
            got = fold_plate(p.plate)
            cut = len(want) >= cfg.MISREAD_PLATE_MIN_PREFIX and got.startswith(want)
            if got != want and (cut or _edits(got, want) <= cfg.MISREAD_PLATE_EDITS):
                how = "cut short" if cut else "one character off"
                cand.misread = f"invoice plate likely misread: {inv.plate} vs photo {p.plate} ({how})"
                cand.veto = None
                return


def score_pair(c: Cluster, inv: InvoiceObs, vpic=None) -> Candidate:
    """``vpic``: VIN -> VpicDecode lookup for the make/model signal (None: skipped)."""
    cand = Candidate(inv.id)
    _vin(c, inv, cand)
    _plate(c, inv, cand)
    _make_model(c, inv, cand, vpic)
    _color(c, inv, cand)
    _date(c, inv, cand)
    _odometer(c, inv, cand)
    _misread(c, inv, cand)
    return cand


# --- Decisions -----------------------------------------------------------------------


@dataclass
class Decision:
    cluster_id: str
    status: str  # linked | review | unmatched
    invoice_id: str | None = None
    decided_by: str | None = None  # vin | plate | score
    score: float | None = None
    candidates: list[Candidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"cluster_id": self.cluster_id, "status": self.status, "invoice_id": self.invoice_id,
                "decided_by": self.decided_by, "score": self.score, "reasons": self.reasons,
                "candidates": [c.to_dict() for c in self.candidates]}


def match(clusters: list[Cluster], invoices: list[InvoiceObs], vpic=None) -> list[Decision]:
    """One decision per cluster, in cluster order. A linked invoice leaves the pool."""
    decisions: dict[str, Decision] = {}
    scored: dict[str, list[Candidate]] = {}
    for c in clusters:
        if c.status == "unassigned":
            decisions[c.id] = Decision(c.id, "review", reasons=[f"photo {c.photos[0].id} {c.reason}"])
            continue
        cands = [score_pair(c, inv, vpic) for inv in invoices]
        scored[c.id] = sorted(cands, key=lambda x: (x.hard is None, -x.score, x.invoice_id))
    linked_by: dict[str, str] = {}  # invoice -> cluster

    def link(cid: str, cand: Candidate, how: str) -> None:
        linked_by[cand.invoice_id] = cid
        decisions[cid] = Decision(cid, "linked", cand.invoice_id, how, cand.score, [cand],
                                  [f"{s.name}: {s.detail}" for s in cand.signals])

    # 1. Hard identifiers.
    for cid, cands in scored.items():
        hard = [x for x in cands if x.hard]
        if not hard:
            continue
        if len(hard) > 1:
            decisions[cid] = Decision(cid, "review", candidates=hard[: cfg.TOP_CANDIDATES],
                                      reasons=["VIN or plate matches several invoices"])
            continue
        [h] = hard
        if h.veto or h.id_conflict:
            decisions[cid] = Decision(cid, "review", candidates=[h], score=h.score,
                                      reasons=[f"{h.hard} matches {h.invoice_id}, but: " + "; ".join(h.conflicts)])
        elif h.invoice_id in linked_by:
            decisions[cid] = Decision(cid, "review", candidates=[h], score=h.score, reasons=[
                f"{h.hard} matches {h.invoice_id}, already linked to cluster {linked_by[h.invoice_id]}"
                " (same vehicle split in two clusters?)"])
        else:
            link(cid, h, h.hard)

    # 2. Scores: best first; the lead counts every non-vetoed invoice, linked or not.
    # A cluster with a likely misread invoice VIN/plate never auto-links on score.
    while True:
        best = None
        for cid, cands in scored.items():
            if cid in decisions or any(x.misread and not x.veto for x in cands):
                continue
            live = [x for x in cands if not x.veto]
            if not live:
                continue
            ranked = sorted(live, key=lambda x: (-x.score, x.invoice_id))
            top = ranked[0]
            lead = top.score - (ranked[1].score if len(ranked) > 1 else 0)
            if (top.score >= cfg.AUTO_SCORE and lead >= cfg.AUTO_LEAD and not top.conflicts
                    and top.invoice_id not in linked_by and (best is None or top.score > best[1].score)):
                best = (cid, top)
        if best is None:
            break
        link(best[0], best[1], "score")

    # 3. Everything else: review or unmatched, with the top candidates still in the pool.
    for cid, cands in scored.items():
        if cid in decisions:
            continue
        vetoed = [x for x in cands if x.veto]
        live = sorted((x for x in cands if not x.veto), key=lambda x: (-x.score, x.invoice_id))
        pool = [x for x in live if x.invoice_id not in linked_by]
        reasons = [f"veto {x.invoice_id}: {x.veto}" for x in vetoed]
        if live and live[0].invoice_id in linked_by:
            reasons.append(f"best candidate {live[0].invoice_id} is already linked to cluster "
                           f"{linked_by[live[0].invoice_id]}")
        top = pool[0].score if pool else 0
        second = pool[1].score if len(pool) > 1 else 0
        misread = [x for x in pool if x.misread]
        if misread:  # those invoices on top, whatever their score
            pool = misread + [x for x in pool if not x.misread]
            decisions[cid] = Decision(cid, "review", score=pool[0].score,
                                      candidates=pool[: max(cfg.TOP_CANDIDATES, len(misread))],
                                      reasons=[x.misread for x in misread] + reasons)
            continue
        if pool and top >= cfg.REVIEW_MIN:
            if top >= cfg.AUTO_SCORE and top - second < cfg.AUTO_LEAD:
                reasons.append(f"top two within {cfg.AUTO_LEAD} points ({top:g} vs {second:g})")
            elif top >= cfg.AUTO_SCORE:
                reasons.append("conflicts: " + "; ".join(pool[0].conflicts))
            else:
                reasons.append(f"score {top:g} below {cfg.AUTO_SCORE}")
            status = "review"
        else:
            reasons.append(f"best score {top:g} below {cfg.REVIEW_MIN}" if pool else "no candidate invoice")
            status = "unmatched"
        decisions[cid] = Decision(cid, status, score=top if pool else None,
                                  candidates=pool[: cfg.TOP_CANDIDATES], reasons=reasons)
    return [decisions[c.id] for c in clusters]
