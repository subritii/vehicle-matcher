"""Restore the scale demo to the state tools/seed_scale_demo.py left it in, in seconds.

Copies data/scale/snapshot.db over the live database in place (an app running on it keeps
working: reload the page), and removes uploads, saved lookups and unsaved photo lookups.

Usage:
    python tools/reset_scale_demo.py [--data data/scale]
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.settings import get_settings  # noqa: E402
from app.storage import LocalStorage  # noqa: E402


def restore(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir).resolve()
    settings = get_settings(data_dir=data_dir, database_url=f"sqlite:///{data_dir / 'app.db'}")  # never a .env DATABASE_URL
    snap = settings.data_dir / "snapshot.db"
    if not snap.exists():
        raise FileNotFoundError(f"{snap} not found: run python tools/seed_scale_demo.py first")
    live = Path(settings.database_url.removeprefix("sqlite:///"))
    src, dst = sqlite3.connect(snap), sqlite3.connect(live)
    with dst:
        src.backup(dst)  # page by page into the live file, so open connections stay valid
    src.close()
    dst.close()
    storage = LocalStorage(settings.media_dir)
    for prefix in ("uploads", "found"):
        storage.delete_prefix(prefix)
    for p in (settings.data_dir / "lookups").glob("*"):
        shutil.rmtree(p, ignore_errors=True)
    return live


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "scale"))
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    try:
        live = restore(args.data)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(f"Scale demo restored in {time.perf_counter() - t0:.1f} s ({live}). If the app is running, reload the page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
