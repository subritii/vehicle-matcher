"""Load the processed dev results into the demo database, so the dashboard opens instantly.

Uses the hard dev invoices (each as pdf, scan or photo in turn, like
tools/match_accuracy.py) and the matching scenario photos. Every engine output comes
from data/extract_cache/ and data/ocr_cache/ (filled by match_accuracy.py), so no OCR
or model runs for known files. One vehicle (--sample) is kept out and copied to
data/media/demo/sample/ for the Upload page.

Usage:
    python tools/seed_demo.py [--model NAME | --no-vision] [--sample inv_057]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import pipeline  # noqa: E402
from app.extract import photo as ph  # noqa: E402
from app.extract.vision import VisionModel  # noqa: E402
from app.match.cluster import cluster_photos  # noqa: E402
from app.match.score import InvoiceObs, match  # noqa: E402
from app.models import Cluster, make_sessionmaker, reset_database  # noqa: E402
from app.services import overview  # noqa: E402
from app.settings import get_settings  # noqa: E402
from app.storage import LocalStorage  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_hard"
RENDERINGS = ("pdf", "scan", "photo")
DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"


def load(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"vision model whose cached answers to use ({DEFAULT_MODEL})")
    ap.add_argument("--no-vision", action="store_true", help="rules + OCR + barcode + fast-alpr only")
    ap.add_argument("--sample", default="inv_057", help="invoice (and its photos) kept out for the Upload page")
    args = ap.parse_args(argv)

    key = FIXTURES / "matching_key.csv"
    if not key.exists():
        print("no matching scenario: run python tools/make_fixtures.py --difficulty hard --matching-only")
        return 1
    photo_rows = load(key)
    invoices = [r for r in load(FIXTURES / "answer_key.csv") if r["split"] == "dev"]
    files = {r["invoice_id"]: FIXTURES / r[RENDERINGS[k % 3]] for k, r in enumerate(invoices)}
    kinds = {r["invoice_id"]: RENDERINGS[k % 3] for k, r in enumerate(invoices)}
    sample_photos = [r for r in photo_rows if r["invoice_id"] == args.sample]
    if args.sample and args.sample not in files:
        print(f"unknown --sample {args.sample}")
        return 1
    photo_rows = [r for r in photo_rows if r["invoice_id"] != args.sample]
    demo_files = {iid: p for iid, p in files.items() if iid != args.sample}

    settings = get_settings()
    reset_database(settings.database_url)  # in place: a running app keeps working
    storage = LocalStorage(settings.media_dir)
    for prefix in ("demo", "uploads", "found"):
        storage.delete_prefix(prefix)
    sessions = make_sessionmaker(settings.database_url)

    vision = None if args.no_vision else VisionModel(args.model)
    readers = pipeline.Readers.build(settings, vision=vision)
    t0 = time.perf_counter()
    print(f"extracting {len(demo_files)} invoices and {len(photo_rows)} photos "
          f"({'rules only' if vision is None else 'vision model ' + args.model}, from the cache)...")
    inv_fields = readers.invoices(demo_files)
    paths = [FIXTURES / r["photo"] for r in photo_rows]
    uploaded = {p: datetime.fromisoformat(r["uploaded_at"]) for p, r in zip(paths, photo_rows)}
    photo_fields = ph.extract_photos(paths, readers.photo_engines(), uploaded_at=uploaded, vpic=readers.vpic)
    print(f"  done in {time.perf_counter() - t0:.0f} s")

    with sessions() as db:
        for iid, path in demo_files.items():
            pipeline.add_invoice(db, storage, iid, path, kinds[iid], inv_fields[iid], readers.ocr_cache, "demo")
        rows = pipeline.add_photos(db, storage, "demo", [
            (pipeline.PhotoIn(path, uploaded[path], r["uploader"]), pf)
            for path, r, pf in zip(paths, photo_rows, photo_fields)])
        set_of = {obs.id: r["set_id"] for (_, obs), r in zip(rows, photo_rows)}
        set_invoice = {r["set_id"]: r["invoice_id"] for r in photo_rows}
        clusters = cluster_photos([obs for _, obs in rows])
        inv_obs = [InvoiceObs.from_invoice(iid, f) for iid, f in inv_fields.items()]
        decisions = match(clusters, inv_obs, vpic=readers.vpic or None)
        truth = {c.id: set_invoice[Counter(set_of[p.id] for p in c.photos).most_common(1)[0][0]] for c in clusters}
        pipeline.apply_match(db, clusters, decisions, readers.vpic, truth=truth)
        db.commit()
        stats = overview(db)
        wrong = [c.id for c in db.query(Cluster).all()
                 if c.status == "linked" and c.truth_invoice_id is not None and c.truth_invoice_id != c.invoice_id]

    # The held-out vehicle for the Upload page.
    if args.sample:
        storage.put_file(f"demo/sample/invoice{files[args.sample].suffix}", files[args.sample])
        for r in sample_photos:
            storage.put_file(f"demo/sample/photos/{Path(r['photo']).name}", FIXTURES / r["photo"])

    print(f"\n{stats['clusters']} photo sets, {stats['photos']} photos, {stats['invoices']} invoices")
    print(f"  auto-linked {stats['auto']}, review {stats['review']}, unmatched {stats['unmatched']}, "
          f"wrong auto-links {stats['wrong_auto']} {wrong if wrong else ''}")
    print(f"  sample for the Upload page: {args.sample} ({kinds[args.sample]}) + {len(sample_photos)} photos")
    print(f"database: {settings.database_url}\nrun: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
