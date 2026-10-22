"""Container start-up (Dockerfile): restore the scale demo, or build it once, then serve it.

- data/snapshot.db exists (a later start): restore it in seconds, like tools/reset_scale_demo.py.
  Every start begins from the same clean demo.
- First start: download the Wikimedia Commons exteriors that exteriors.csv lists (if they
  aren't in the image or volume yet; needs WIKIMEDIA_CONTACT), then build the scale demo
  with tools/seed_scale_demo.py and write the photos of the cars on the lot to the photos
  folder (bind-mounted to ./demo_photos on the host).

Then it runs uvicorn on the scale demo database. Settings (environment):
    DATA_DIR (/app/data/scale), DEMO_PHOTOS_DIR (/app/demo_photos), SCALE_COUNT (200),
    WIKIMEDIA_CONTACT, PORT (8000)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def prepare(data: Path, photos: Path, count: int, contact: str, restore, seed, fetch, have_exteriors) -> str:
    """'restored' or 'built'; exits with a message when the demo can't be built."""
    data = Path(data)
    if (data / "snapshot.db").exists():
        restore(data)
        return "restored"
    if not have_exteriors():
        if not contact:
            raise SystemExit("The first start downloads car photos from Wikimedia Commons, which asks for contact "
                             "details: set WIKIMEDIA_CONTACT to your email or a URL (see the README).")
        if fetch(contact):
            raise SystemExit("Downloading the Wikimedia Commons photos failed (see above).")
    data.mkdir(parents=True, exist_ok=True)
    if seed(["--count", str(count), "--data", str(data), "--photos", str(photos)]):
        raise SystemExit("Building the scale demo failed (see above).")
    return "built"


def server(data: Path, port: int = 8000, base_env: dict | None = None) -> tuple[list[str], dict]:
    """The uvicorn command and environment for the scale demo. Compose publishes the port on
    127.0.0.1 only; requests arrive from Docker's private gateway address, so the app's
    local-network rule needs no change."""
    env = dict(os.environ if base_env is None else base_env)
    env["DATA_DIR"] = str(data)
    env["DATABASE_URL"] = f"sqlite:///{Path(data) / 'app.db'}"
    return ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)], env


def _fetch(contact: str) -> int:
    import fetch_exteriors

    os.environ["WIKIMEDIA_CONTACT"] = contact
    return fetch_exteriors.main(["--listed"])


def _have_exteriors() -> bool:
    import make_fixtures

    return bool(make_fixtures.load_exteriors())


def main() -> int:
    import reset_scale_demo
    import seed_scale_demo

    data = Path(os.getenv("DATA_DIR") or ROOT / "data" / "scale").resolve()
    photos = Path(os.getenv("DEMO_PHOTOS_DIR") or ROOT / "demo_photos")
    count = int(os.getenv("SCALE_COUNT") or 200)
    contact = os.getenv("WIKIMEDIA_CONTACT", "").strip()
    how = prepare(data, photos, count, contact, restore=reset_scale_demo.restore, seed=seed_scale_demo.main,
                  fetch=_fetch, have_exteriors=_have_exteriors)
    print(f"Scale demo {how} in {data}. Photos of the cars on the lot: {photos} (./demo_photos on your computer).",
          flush=True)
    argv, env = server(data, int(os.getenv("PORT") or 8000))
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    raise SystemExit(main())
