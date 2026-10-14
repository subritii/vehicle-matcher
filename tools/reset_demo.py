"""Reset the demo to its starting state, e.g. between takes of a screen recording.

Re-runs tools/seed_demo.py: the database is emptied in place and reloaded from the
cached dev results (~15 s, no OCR or model), and uploads, saved lookups and the held-out
sample are restored. A running app keeps working: reload the page.

Usage:
    python tools/reset_demo.py [--model NAME | --no-vision] [--sample inv_057]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import seed_demo  # noqa: E402

from app.settings import get_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    rc = seed_demo.main(argv)
    if rc == 0:
        lookups = get_settings().data_dir / "lookups"
        for p in lookups.glob("*") if lookups.exists() else []:
            shutil.rmtree(p, ignore_errors=True)  # unsaved photo lookups
        print("\nDemo reset. If the app is running, just reload the page.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
