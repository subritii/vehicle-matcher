"""Showcase mode (DEMO_MODE=showcase): the free public demo, read-only and under 512 MB.

It serves a bundle that tools/build_showcase.py made on a computer with the full pipeline:
- app.db: the processed demo database, opened read-only and shared by every visitor;
- media/: only the images that database references (no uploads, no held-out sample);
- samples.json and samples/<id>/thumb_NN.jpg: photo lookups run by the real pipeline (OCR,
  plate reader, vision model), saved with their "How it found this" details;
- credits.json: the Wikimedia Commons photos in the bundle, with authors and licenses.

Nothing heavy is loaded: no OCR, plate reader or vision model. A visitor's review decisions
go to a private copy of the database, made on their first decision and dropped after two
idle hours (or when too many visitors are active), so the shared database never changes.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.lookup import Lookup
from app.models import make_sessionmaker

DB_FILE = "app.db"
SAMPLES_FILE = "samples.json"
CREDITS_FILE = "credits.json"
COOKIE = "vm_visitor"
VISITOR_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
VISITOR_IDLE = timedelta(hours=2)
MAX_VISITORS = 200  # session databases kept at once (about 0.5 MB each)
MAX_BODY = 16 * 1024  # the only POSTs left are review decisions: a few form fields
REVIEW_POST = re.compile(r"^/review/[^/]+/(confirm|pick|no-invoice|undo)$")
DEFAULT_PAGE_LIMIT = 120  # requests per minute per visitor (images and static files not counted)
DEFAULT_POST_LIMIT = 20


# --- The bundle --------------------------------------------------------------------------


@dataclass
class Sample:
    """One saved photo lookup: the photos' names, the result and how long it took."""

    id: str
    car: str
    label: str
    names: list[str]
    seconds: float
    result: dict
    dir: Path
    created: datetime

    def lookup(self) -> Lookup:
        """A finished Lookup, so the Find templates show it like a live one."""
        item = Lookup(self.id, self.dir, list(self.names), created=self.created, status="done", result=self.result)
        for s in item.steps:
            s["status"] = "done"
        return item


@dataclass
class Bundle:
    dir: Path
    meta: dict
    cars: list[dict]
    samples: dict[str, Sample]
    credits: list[dict]

    @property
    def db_path(self) -> Path:
        return self.dir / DB_FILE

    @property
    def media_dir(self) -> Path:
        return self.dir / "media"

    @classmethod
    def load(cls, folder: str | Path) -> Bundle:
        d = Path(folder)
        if not (d / DB_FILE).exists() or not (d / SAMPLES_FILE).exists():
            raise FileNotFoundError(f"no showcase bundle in {d}: build it with python tools/build_showcase.py")
        data = json.loads((d / SAMPLES_FILE).read_text())
        meta = data.get("meta", {})
        built = datetime.fromisoformat(meta["built_at"]) if meta.get("built_at") else datetime.now()
        samples = {s["id"]: Sample(s["id"], s["car"], s["label"], s["names"], s.get("seconds", 0.0), s["result"],
                                   d / "samples" / s["id"], built) for s in data["samples"]}
        credits = json.loads((d / CREDITS_FILE).read_text()) if (d / CREDITS_FILE).exists() else []
        return cls(d, meta, data.get("cars", []), samples, credits)

    def gallery(self) -> list[dict]:
        """Per car: its single-photo samples, and the sample with all its photos."""
        out = []
        for car in self.cars:
            mine = [s for s in self.samples.values() if s.car == car["id"]]
            out.append({"car": car, "photos": [s for s in mine if len(s.names) == 1],
                        "all": next((s for s in mine if len(s.names) > 1), None)})
        return out


# --- One shared read-only database, one private copy per deciding visitor ----------------


def read_only_sessionmaker(path: Path) -> sessionmaker:
    from urllib.parse import quote

    uri = f"file:{quote(str(Path(path).resolve()))}?mode=ro&immutable=1"
    engine = create_engine("sqlite://", poolclass=QueuePool,
                           creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False))
    return sessionmaker(engine, expire_on_commit=False)


class VisitorDatabases:
    def __init__(self, base_db: Path, state_dir: Path, max_visitors: int = MAX_VISITORS,
                 idle: timedelta = VISITOR_IDLE) -> None:
        self.base_db, self.dir = Path(base_db), Path(state_dir)
        self.max_visitors, self.idle = max_visitors, idle
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob("*.db*"):  # sessions of an earlier run
            old.unlink(missing_ok=True)
        self.base = read_only_sessionmaker(self.base_db)
        self._items: OrderedDict[str, tuple[sessionmaker, datetime]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, vid: str | None) -> sessionmaker | None:
        if not vid or not VISITOR_ID.match(vid):
            return None
        with self._lock:
            entry = self._items.get(vid)
            if entry is None:
                return None
            self._items[vid] = (entry[0], datetime.now())
            self._items.move_to_end(vid)
            return entry[0]

    def create(self) -> str:
        self.cleanup()
        vid = secrets.token_urlsafe(18)
        path = self.dir / f"{vid}.db"
        shutil.copyfile(self.base_db, path)
        sm = make_sessionmaker(f"sqlite:///{path}")
        dropped = []
        with self._lock:
            self._items[vid] = (sm, datetime.now())
            while len(self._items) > self.max_visitors:
                dropped.append(self._items.popitem(last=False))
        for old, (old_sm, _) in dropped:
            self._drop(old, old_sm)
        return vid

    def cleanup(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        with self._lock:
            old = [(vid, sm) for vid, (sm, seen) in self._items.items() if now - seen > self.idle]
            for vid, _ in old:
                del self._items[vid]
        for vid, sm in old:
            self._drop(vid, sm)

    def _drop(self, vid: str, sm: sessionmaker) -> None:
        sm.kw["bind"].dispose()
        for suffix in ("", "-wal", "-shm"):
            (self.dir / f"{vid}.db{suffix}").unlink(missing_ok=True)


current_visitor: ContextVar[sessionmaker | None] = ContextVar("showcase_visitor", default=None)


class Sessions:
    """Stands in for the app's sessionmaker: the visitor's own copy if they have one, else
    the shared read-only database."""

    def __init__(self, visitors: VisitorDatabases) -> None:
        self.visitors = visitors

    def __call__(self):
        return (current_visitor.get() or self.visitors.base)()


# --- Rate limiting -------------------------------------------------------------------------


class RateLimiter:
    """At most ``limit`` requests per ``window`` seconds per key (a sliding window)."""

    def __init__(self, limit: int, window: float = 60.0, max_keys: int = 10_000) -> None:
        self.limit, self.window, self.max_keys = limit, window, max_keys
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > self.max_keys:
                self._hits = {k: q for k, q in self._hits.items() if q and q[-1] > now - self.window}
            return True


def limit_from_env(name: str, default: int) -> int:
    import os

    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else default


def counted(path: str) -> bool:
    """Pages and actions count toward the rate limit; the images and files a page loads don't."""
    return not path.startswith(("/static/", "/media/")) and not path.endswith(".jpg")
