"""Decisions (link, no invoice) and the queries behind the pages."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import func, select

from app.models import Cluster, Invoice, MatchEvent, Photo, Vehicle


class ActionError(Exception):
    """A review action that cannot be applied (already decided, invoice taken...)."""


# --- Decisions --------------------------------------------------------------------------


def link_cluster(db, cluster: Cluster, invoice: Invoice | None, *, decided_by: str, action: str,
                 how: str | None = None, score: float | None = None, signals: list | None = None) -> Vehicle:
    """Link a cluster to an invoice (one-to-one) and record the vehicle and a match event."""
    if invoice is None:
        raise ActionError("invoice not found")
    if invoice.status == "linked":
        raise ActionError(f"invoice {invoice.id} is already linked to another photo set")
    if cluster.status == "linked":
        raise ActionError(f"photo set {cluster.id} is already linked")
    s, o = cluster.summary_json or {}, invoice.obs_json or {}
    if s.get("vin_trusted"):
        vin = s["vin"]  # a trusted photo VIN beats an untrusted (maybe misread) invoice VIN
    elif o.get("vin_trusted") or not s.get("vin"):
        vin = o.get("vin")
    else:
        vin = s.get("vin")
    year = o.get("year") or s.get("year")
    vehicle = Vehicle(vin=vin, plate=s.get("plate") if s.get("plate_trusted") else (o.get("plate") or s.get("plate")),
                      state=s.get("plate_state") or o.get("plate_state"),
                      year=int(year) if str(year or "").isdigit() else None,
                      make=o.get("make") or s.get("make"), model=o.get("model") or s.get("model"),
                      color=o.get("color") or s.get("color"))
    db.add(vehicle)
    db.flush()
    cluster.status, cluster.invoice_id, cluster.vehicle_id = "linked", invoice.id, vehicle.id
    cluster.decided_by, cluster.how = decided_by, how
    if score is not None:
        cluster.score = score
    invoice.status, invoice.vehicle_id = "linked", vehicle.id
    db.add(MatchEvent(cluster_id=cluster.id, invoice_id=invoice.id, action=action, score=score,
                      signals_json=signals or [], decided_by=decided_by))
    return vehicle


def _reviewable(db, cid: str) -> Cluster:
    cluster = db.get(Cluster, cid)
    if cluster is None:
        raise ActionError("photo set not found")
    if cluster.status != "review":
        raise ActionError(f"photo set {cid} is not waiting for review (status: {cluster.status})")
    return cluster


def review_link(db, cid: str, invoice_id: str, action: str) -> Cluster:
    """Reviewer links ``cid`` to ``invoice_id`` (action: confirm | pick)."""
    cluster = _reviewable(db, cid)
    cand = next((c for c in cluster.candidates_json if c["invoice_id"] == invoice_id), None)
    link_cluster(db, cluster, db.get(Invoice, invoice_id), decided_by="user", action=action,
                 score=cand["score"] if cand else None, signals=cand["signals"] if cand else [])
    return cluster


def review_no_invoice(db, cid: str) -> Cluster:
    cluster = _reviewable(db, cid)
    cluster.status, cluster.decided_by = "no_invoice", "user"
    db.add(MatchEvent(cluster_id=cid, invoice_id=None, action="no_invoice", decided_by="user"))
    return cluster


UNDOABLE = ("confirm", "pick", "no_invoice")


def last_undoable(db, cid: str) -> MatchEvent | None:
    """The cluster's latest event, if it is a reviewer decision that can be undone."""
    last = db.scalar(select(MatchEvent).where(MatchEvent.cluster_id == cid).order_by(MatchEvent.id.desc()).limit(1))
    return last if last is not None and last.decided_by == "user" and last.action in UNDOABLE else None


def review_undo(db, cid: str) -> Cluster:
    """Put a reviewed photo set back in the queue. History stays: an "undo" event is added;
    only the vehicle record the link created is removed."""
    cluster = db.get(Cluster, cid)
    if cluster is None:
        raise ActionError("photo set not found")
    ev = last_undoable(db, cid)
    if ev is None:
        raise ActionError(f"photo set {cid} has no reviewer decision to undo")
    if ev.action in ("confirm", "pick"):
        inv, vehicle_id = db.get(Invoice, cluster.invoice_id), cluster.vehicle_id
        inv.status, inv.vehicle_id = "unlinked", None
        cluster.invoice_id = cluster.vehicle_id = None
        db.flush()
        if vehicle_id is not None:
            db.delete(db.get(Vehicle, vehicle_id))
    cluster.status, cluster.decided_by, cluster.how = "review", None, None
    db.add(MatchEvent(cluster_id=cid, invoice_id=ev.invoice_id, action="undo", decided_by="user"))
    return cluster


def undo_banner(db, cid: str | None) -> dict | None:
    """What the reviewer just decided, for the Undo banner (None if it can't be undone)."""
    ev = last_undoable(db, cid) if cid else None
    if ev is None:
        return None
    text = (f"Marked photo set {cid} as having no invoice." if ev.action == "no_invoice"
            else f"Linked photo set {cid} to invoice {ev.invoice_id}.")
    return {"cluster_id": cid, "text": text}


# --- Queries ----------------------------------------------------------------------------


def review_queue(db) -> list[Cluster]:
    """Pending review items: those with candidate invoices first."""
    rows = db.scalars(select(Cluster).where(Cluster.status == "review").order_by(Cluster.id)).all()
    return sorted(rows, key=lambda c: (not c.candidates_json, c.id))


def review_count(db) -> int:
    return db.scalar(select(func.count()).select_from(Cluster).where(Cluster.status == "review")) or 0


def overview(db) -> dict:
    clusters = db.scalars(select(Cluster)).all()
    n = len(clusters)
    auto = [c for c in clusters if c.status == "linked" and c.decided_by == "auto"]
    # truth_invoice_id: None = unknown (uploads), "" = known to have no invoice (demo answer key)
    checked = [c for c in auto if c.truth_invoice_id is not None]
    wrong = [c for c in checked if c.truth_invoice_id != c.invoice_id]
    count = {s: sum(1 for c in clusters if c.status == s) for s in ("review", "unmatched", "no_invoice")}
    user = sum(1 for c in clusters if c.status == "linked" and c.decided_by == "user")
    return {
        "clusters": n,
        "photos": db.scalar(select(func.count()).select_from(Photo)) or 0,
        "invoices": db.scalar(select(func.count()).select_from(Invoice)) or 0,
        "unlinked_invoices": db.scalar(select(func.count()).select_from(Invoice).where(Invoice.status != "linked")),
        "auto": len(auto), "user": user, "review": count["review"], "unmatched": count["unmatched"],
        "no_invoice": count["no_invoice"], "wrong_auto": len(wrong), "checked": len(checked),
        "pct": (lambda k: round(100 * k / n) if n else 0),
    }


def linked_vehicles(db) -> list[Cluster]:
    return db.scalars(select(Cluster).where(Cluster.status == "linked").order_by(Cluster.invoice_id)).all()


def invoices_without_photos(db, limit: int | None = None) -> list[Invoice]:
    """Newest first."""
    q = select(Invoice).where(Invoice.status != "linked").order_by(Invoice.date.desc(), Invoice.id)
    return db.scalars(q.limit(limit) if limit else q).all()


def sets_without_invoice(db) -> list[Cluster]:
    return db.scalars(select(Cluster).where(Cluster.status.in_(("unmatched", "no_invoice")))
                      .order_by(Cluster.id)).all()


def unlinked_invoices(db) -> list[Invoice]:
    return invoices_without_photos(db)


def events_for(db, cid: str) -> list[MatchEvent]:
    return db.scalars(select(MatchEvent).where(MatchEvent.cluster_id == cid).order_by(MatchEvent.id)).all()


# --- Review comparison ------------------------------------------------------------------


def _fmt_date(v) -> str | None:
    if not v:
        return None
    if isinstance(v, (date, datetime)):
        return v.strftime("%b %d, %Y")
    try:
        return datetime.fromisoformat(str(v)).strftime("%b %d, %Y")
    except ValueError:
        return str(v)


def _join(*parts) -> str | None:
    s = " ".join(str(p) for p in parts if p not in (None, ""))
    return s or None


def compare_rows(cluster: Cluster, invoice: Invoice, cand: dict) -> list[dict]:
    """Field by field: what the photos say vs what the invoice says, and whether the
    matcher counted it as the same (agree), different (conflict) or couldn't compare (none)."""
    s = cluster.summary_json or {}
    signals = {x["name"]: x for x in cand.get("signals", [])}
    conflicts = cand.get("conflicts", [])
    misread = cand.get("misread") or ""

    def state(signal: str, *prefixes: str) -> str:
        if signal in signals and signals[signal]["points"] > 0:
            return "agree"
        if any(c.startswith(p) for c in conflicts for p in prefixes):
            return "conflict"
        return "none"

    f = lambda name: invoice.field(name)  # noqa: E731
    make = s.get("make")
    rows = [
        {"key": "vin", "label": "VIN", "photo": s.get("vin"), "invoice": f("vin"), "state": state("vin", "VIN"),
         "mono": True, "photo_note": "confirmed" if s.get("vin_trusted") else ("read once" if s.get("vin") else ""),
         "note": "probably misread on the invoice" if misread.startswith("invoice VIN") else ""},
        {"key": "plate", "label": "License plate", "photo": _join(s.get("plate"), s.get("plate_state")),
         "invoice": _join(f("plate"), f("plate_state")), "state": state("plate", "plate"), "mono": True,
         "photo_note": "confirmed" if s.get("plate_trusted") else "",
         "note": "probably misread on the invoice" if misread.startswith("invoice plate") else ""},
        {"key": "make_model", "label": "Make & model",
         "photo": _join(s.get("year"), make.title() if make and len(make) > 3 else make, s.get("model")),
         "invoice": _join(f("year"), f("make"), f("model")), "state": state("make_model", "make/model"),
         "photo_note": "from the VIN" if make else ""},
        {"key": "color", "label": "Color", "photo": (s.get("color") or "").capitalize() or None,
         "invoice": (f("color") or "").capitalize() or None, "state": state("color", "color")},
        {"key": "odometer", "label": "Mileage", "photo": f"{s['odometer']:,} mi" if s.get("odometer") is not None else None,
         "invoice": f"{int(f('mileage')):,} mi" if str(f("mileage") or "").isdigit() else f("mileage"),
         "state": state("odometer", "odometer")},
        {"key": "date", "label": "Date", "photo": _fmt_date(s.get("time")), "invoice": _fmt_date(f("invoice_date")),
         "state": state("date"), "photo_note": "" if s.get("time_from_exif") or not s.get("time") else "upload date"},
    ]
    for r in rows:
        r["points"] = signals[r["key"]]["points"] if r["key"] in signals else 0
        if r["state"] == "agree":
            r["word"] = "Close" if r["key"] == "date" else "Same"
        elif r["state"] == "conflict":
            r["word"] = "Different"
        elif r["photo"] and r["invoice"]:
            r["word"] = "Far apart" if r["key"] == "date" else "Not compared"
        else:
            r["word"] = "Not on invoice" if r["photo"] else ("Not in photos" if r["invoice"] else "Not available")
    return rows


# --- Plain language -----------------------------------------------------------------------

_PLAIN = (
    (r"invoice VIN likely misread: (\S+) vs photo (\S+) \((\d+) differences?\)",
     lambda m: f"The VIN on the invoice is {m[3]} characters different from the VIN in the photos. "
               "It was probably misread from the invoice."),
    (r"invoice plate likely misread: (\S+) vs photo (\S+) \(cut short\)",
     lambda m: f"The license plate on the invoice ({m[1]}) looks cut short. The photos show {m[2]}."),
    (r"invoice plate likely misread: (\S+) vs photo (\S+) \(.*\)",
     lambda m: f"The license plate on the invoice ({m[1]}) is one character off from the photos ({m[2]})."),
    (r"(?:best )?score [\d.]+ below 80", lambda m: "Some details match, but not enough to be sure it's the same car."),
    (r"best score [\d.]+ below 40", lambda m: "No invoice is close enough."),
    (r"top two within", lambda m: "Two invoices fit about equally well."),
    (r"conflicts: ", lambda m: "The best invoice agrees on most details but disagrees on one."),
    (r"best candidate \S+ is already linked", lambda m: "The best-fitting invoice already has photos of another car."),
    (r"This photo fits two clusters", lambda m: "This photo could belong to either of two cars photographed minutes apart."),
    (r"fits two clusters", lambda m: "This photo could belong to either of two cars photographed minutes apart."),
    (r"VIN or plate matches several invoices", lambda m: "The VIN or plate appears on more than one invoice."),
    (r"(vin|plate) matches \S+, already linked",
     lambda m: "The VIN matches an invoice that already has photos. These may be more photos of the same car."),
    (r"(vin|plate) matches \S+, but", lambda m: f"The {'VIN' if m[1] == 'vin' else 'plate'} matches, but another detail disagrees."),
    (r"no candidate invoice", lambda m: "There are no invoices to compare with yet."),
    (r"saved from Find invoice from photo", lambda m: "Photos added with Find invoice by photo."),
)


def plain_reason(text: str) -> str:
    """A matcher reason in words a service manager understands."""
    import re

    for pattern, say in _PLAIN:
        m = re.search(pattern, text)
        if m:
            return say(m)
    return text


def plain_reasons(reasons: list[str]) -> list[str]:
    out = [plain_reason(r) for r in reasons if not r.startswith("veto ")]
    vetoes = sum(r.startswith("veto ") for r in reasons)
    if vetoes:
        out.append(f"{vetoes} other invoice{'s were' if vetoes != 1 else ' was'} ruled out because the VIN or "
                   "plate is different.")
    return list(dict.fromkeys(out))


def match_sentence(cluster) -> str:
    """One sentence on why a photo set is matched to its invoice."""
    if cluster.decided_by == "user":
        events = cluster.reasons_json or []
        if any("Find invoice" in r for r in events):
            return "Matched with Find invoice by photo: a person saved these photos to this invoice."
        return "Matched after a check: a person confirmed these photos belong to this invoice."
    return {
        "vin": "Matched automatically: the VIN on the invoice and the VIN in the photos are identical.",
        "plate": "Matched automatically: the license plate and state on the invoice and in the photos are identical.",
        "score": "Matched automatically: make, model, color, mileage and date agree, and no other invoice comes close.",
    }.get(cluster.how or "", "Matched automatically.")
