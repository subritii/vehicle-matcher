"""SQLAlchemy models (vehicles, invoices, photos, clusters, match events) plus upload jobs. SQLite for the MVP; nothing
here is SQLite-specific."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


def now() -> datetime:
    return datetime.now().replace(microsecond=0)


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(primary_key=True)
    vin: Mapped[str | None] = mapped_column(String(17), index=True)
    plate: Mapped[str | None] = mapped_column(String(12))
    state: Mapped[str | None] = mapped_column(String(2))
    year: Mapped[int | None]
    make: Mapped[str | None] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(60))
    color: Mapped[str | None] = mapped_column(String(20))

    @property
    def title(self) -> str:
        return " ".join(str(x) for x in (self.year, self.make, self.model) if x) or "Unknown vehicle"


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    vehicle_id: Mapped[int | None] = mapped_column(ForeignKey("vehicles.id"))
    vendor: Mapped[str | None] = mapped_column(String(120))
    invoice_no: Mapped[str | None] = mapped_column(String(40))
    date: Mapped[date | None] = mapped_column(Date, index=True)
    mileage: Mapped[int | None] = mapped_column(Integer, index=True)
    total: Mapped[str | None] = mapped_column(String(20))
    # Lookup keys (index_keys): VIN, its four pieces (a VIN a few characters off still shares
    # one), the plate with O/0 and I/1 folded, make, model and color.
    vin: Mapped[str | None] = mapped_column(String(20), index=True)
    vin_p1: Mapped[str | None] = mapped_column(String(5), index=True)
    vin_p2: Mapped[str | None] = mapped_column(String(4), index=True)
    vin_p3: Mapped[str | None] = mapped_column(String(4), index=True)
    vin_p4: Mapped[str | None] = mapped_column(String(8), index=True)
    plate_key: Mapped[str | None] = mapped_column(String(12), index=True)
    make_key: Mapped[str | None] = mapped_column(String(40), index=True)
    model_key: Mapped[str | None] = mapped_column(String(60))
    color: Mapped[str | None] = mapped_column(String(20), index=True)
    file_path: Mapped[str] = mapped_column(String(300))  # original, under media/
    image_path: Mapped[str | None] = mapped_column(String(300))  # display image, under media/
    file_kind: Mapped[str] = mapped_column(String(10), default="pdf")  # pdf | scan | photo
    extracted_json: Mapped[dict] = mapped_column(JSON, default=dict)  # InvoiceFields.to_dict + boxes
    obs_json: Mapped[dict] = mapped_column(JSON, default=dict)  # app.match.score.InvoiceObs
    status: Mapped[str] = mapped_column(String(12), default="unlinked")  # unlinked | linked
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    vehicle: Mapped[Vehicle | None] = relationship()

    def field(self, name: str):
        return (self.extracted_json.get("fields") or {}).get(name, {}).get("value")


VIN_PIECES = ((0, 5), (5, 9), (9, 13), (13, 20))  # 5+4+4+4: three substitutions leave one piece intact


def vin_pieces(vin: str) -> list[str]:
    return [vin[a:b] for a, b in VIN_PIECES]


def index_keys(inv: Invoice) -> Invoice:
    """Fill the lookup columns from the invoice observation (obs_json)."""
    from app.match.cluster import fold_plate

    o = inv.obs_json or {}
    vin = (o.get("vin") or "").upper() or None
    inv.vin = vin
    inv.vin_p1, inv.vin_p2, inv.vin_p3, inv.vin_p4 = vin_pieces(vin) if vin else (None,) * 4
    inv.plate_key = fold_plate(o["plate"]) if o.get("plate") else None
    inv.make_key = (o.get("make") or "").lower() or None
    inv.model_key = (o.get("model") or "").lower() or None
    inv.color = (o.get("color") or "").lower() or None
    if inv.mileage is None and isinstance(o.get("mileage"), int):
        inv.mileage = o["mileage"]
    return inv


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    vehicle_id: Mapped[int | None] = mapped_column(ForeignKey("vehicles.id"))
    invoice_id: Mapped[str | None] = mapped_column(ForeignKey("invoices.id"))
    status: Mapped[str] = mapped_column(String(12))  # linked | review | unmatched | no_invoice
    decided_by: Mapped[str | None] = mapped_column(String(10))  # auto | user
    how: Mapped[str | None] = mapped_column(String(10))  # vin | plate | score (auto links)
    score: Mapped[float | None] = mapped_column(Float)
    candidates_json: Mapped[list] = mapped_column(JSON, default=list)  # Candidate.to_dict, best first
    reasons_json: Mapped[list] = mapped_column(JSON, default=list)
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)  # VIN, plate, color... read from photos
    truth_invoice_id: Mapped[str | None] = mapped_column(String(40))  # demo answer key; "" = no invoice
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    photos: Mapped[list[Photo]] = relationship(back_populates="cluster", order_by="Photo.sort_key")
    invoice: Mapped[Invoice | None] = relationship()
    vehicle: Mapped[Vehicle | None] = relationship()


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    cluster_id: Mapped[str | None] = mapped_column(ForeignKey("clusters.id"))
    file_path: Mapped[str] = mapped_column(String(300))  # display image, under media/
    thumb_path: Mapped[str] = mapped_column(String(300))
    shot_type: Mapped[str | None] = mapped_column(String(12))
    captured_at: Mapped[datetime | None] = mapped_column(DateTime)
    time_from_exif: Mapped[bool] = mapped_column(default=False)
    gps: Mapped[list | None] = mapped_column(JSON)
    uploader: Mapped[str | None] = mapped_column(String(60))
    read_vin: Mapped[str | None] = mapped_column(String(17))
    read_plate: Mapped[str | None] = mapped_column(String(12))
    read_odometer: Mapped[int | None]
    extracted_json: Mapped[dict] = mapped_column(JSON, default=dict)  # PhotoFields.to_dict
    obs_json: Mapped[dict] = mapped_column(JSON, default=dict)  # app.match.cluster.PhotoObs
    sort_key: Mapped[int] = mapped_column(Integer, default=9)  # VIN/plate/odometer shots first
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    cluster: Mapped[Cluster | None] = relationship(back_populates="photos")

    def source(self, name: str) -> str:
        return (self.extracted_json.get("fields") or {}).get(name, {}).get("source", "")


class MatchEvent(Base):
    __tablename__ = "match_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    cluster_id: Mapped[str] = mapped_column(ForeignKey("clusters.id"), index=True)
    invoice_id: Mapped[str | None] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(12))  # linked | review | unmatched | confirm | pick | no_invoice
    score: Mapped[float | None] = mapped_column(Float)
    signals_json: Mapped[list] = mapped_column(JSON, default=list)
    decided_by: Mapped[str] = mapped_column(String(10))  # auto | user
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Job(Base):
    """An upload processed in the background, one step at a time."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(10), default="queued")  # queued | running | done | failed
    steps_json: Mapped[list] = mapped_column(JSON, default=list)  # [{key, label, status, detail}]
    invoice_path: Mapped[str] = mapped_column(String(300))
    photo_paths: Mapped[list] = mapped_column(JSON, default=list)
    uploader: Mapped[str | None] = mapped_column(String(60))
    invoice_id: Mapped[str | None] = mapped_column(String(40))
    cluster_ids: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


def _engine(url: str):
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _pragmas(conn, _):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

    return engine


@event.listens_for(Invoice, "before_insert")
@event.listens_for(Invoice, "before_update")
def _keep_keys(mapper, connection, target: Invoice) -> None:
    index_keys(target)  # the lookup columns always follow obs_json


def _migrate(engine) -> None:
    """Databases made before the lookup columns: add them, their indexes, and fill them."""
    from sqlalchemy import inspect, text

    have = {c["name"] for c in inspect(engine).get_columns("invoices")}
    missing = [c for c in Invoice.__table__.columns if c.name not in have]
    if missing:
        with engine.begin() as conn:
            for col in missing:
                conn.execute(text(f"ALTER TABLE invoices ADD COLUMN {col.name} {col.type.compile(engine.dialect)}"))
    for ix in Invoice.__table__.indexes:
        ix.create(engine, checkfirst=True)
    if missing:
        with sessionmaker(engine)() as db:
            for inv in db.query(Invoice):
                index_keys(inv)
            db.commit()


def make_sessionmaker(url: str) -> sessionmaker:
    engine = _engine(url)
    Base.metadata.create_all(engine)
    _migrate(engine)
    return sessionmaker(engine, expire_on_commit=False)


def reset_database(url: str) -> None:
    """Empty the database in place (drop and recreate the tables). Unlike deleting the
    file, an app that is running on it keeps working and sees the new data."""
    engine = _engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()
