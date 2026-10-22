"""Build the showcase bundle (showcase/) that DEMO_MODE=showcase serves as the free live demo.

Runs on a computer with the full pipeline (OCR stack, and Ollama for the vision model):
1. Seeds a fresh copy of the demo database (the hard dev invoices and photo sets, from the
   engine caches like tools/seed_demo.py) in data/showcase_build/. The regular demo in data/
   is not touched.
2. Draws new photos of four cars (VIN sticker, plate, odometer, Commons exterior): two whose
   invoice has no photos yet, one already linked, one with no invoice. They are made with a
   seed of their own, so they are neither the demo's photo sets nor ~/Desktop/demo_photos.
3. Looks each photo up, and each car's photos together, with the app's own lookup code and
   engines, and saves the results with their "How it found this" details and timing.
4. Writes showcase/: the database (read-only, no upload jobs), only the media it references,
   samples.json with the saved lookups and their thumbnails, and credits.json for the
   Wikimedia Commons photos.

Usage (ollama serve running, nothing else heavy: OCR and the model take turns):
    python tools/build_showcase.py [--model qwen3.5:4b-q4_K_M | --no-vision]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sqlite3
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import lookup  # noqa: E402
from app.showcase import CREDITS_FILE, DB_FILE, SAMPLES_FILE  # noqa: E402

OUT = ROOT / "showcase"
BUILD = ROOT / "data" / "showcase_build"
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_hard"
DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
SIDE = 1200  # long side of the bundled photos and invoice images
SEED = "showcase-2026"
DEMO_SAMPLE = "inv_057"  # tools/seed_demo.py keeps this vehicle out of the database
KINDS = {"VIN": "VIN sticker", "plate": "License plate", "odometer": "Odometer", "exterior": "Outside"}
# (car, invoice it belongs to or None, shots, note shown in the gallery)
CARS = (
    ("car1", "inv_001", ("VIN", "plate", "odometer", "exterior"), "Its invoice is in the system, without photos yet."),
    ("car2", "inv_046", ("VIN", "exterior"), "Back for a second visit: its invoice already has photos."),
    ("car3", "inv_041", ("plate", "odometer"), "Its invoice is in the system; no VIN photo this time."),
    ("car4", None, ("VIN", "exterior"), "A car the shop has no invoice for."),
)
NOT_PUBLIC = ("sample", "uploads", "found", "lookups", "private")  # media folders never bundled


# --- Lookups ------------------------------------------------------------------------------


def run_sample(sessions, read_photos, sid: str, car: str, label: str, photos: list[Path], vision_on: bool,
               work: Path) -> dict:
    """One lookup with the app's own code, timed; the result as plain JSON."""
    item = lookup.Lookup(sid, Path(work) / sid, [p.name for p in photos])
    shutil.rmtree(item.dir, ignore_errors=True)
    item.dir.mkdir(parents=True)
    for src, dst in zip(photos, item.files):
        shutil.copyfile(src, dst)
    t0 = time.perf_counter()
    lookup.run_lookup(item, read_photos, sessions, vision_on)
    seconds = time.perf_counter() - t0
    if item.status != "done":
        raise RuntimeError(f"lookup {sid} failed: {item.error}")
    result = json.loads(json.dumps(item.result, default=str))
    return {"id": sid, "car": car, "label": label, "names": list(item.names), "seconds": round(seconds, 1),
            "result": result, "thumbs": [item.thumb(i) for i in range(len(photos))]}


# --- The bundle ----------------------------------------------------------------------------


def finalize_db(src: Path, dst: Path) -> None:
    """A consistent copy without upload jobs, in rollback-journal mode so it opens read-only."""
    a, b = sqlite3.connect(src), sqlite3.connect(dst)
    with b:
        a.backup(b)
    a.close()
    b.execute("DELETE FROM jobs")
    b.commit()
    b.execute("PRAGMA journal_mode=DELETE")
    b.execute("VACUUM")
    b.close()


def referenced_media(db_file: Path) -> set[str]:
    con = sqlite3.connect(db_file)
    try:
        keys = {k for row in con.execute("SELECT file_path, image_path FROM invoices") for k in row if k}
        keys |= {k for row in con.execute("SELECT file_path, thumb_path FROM photos") for k in row if k}
    finally:
        con.close()
    return keys


def publishable(key: str) -> bool:
    parts = Path(key).parts
    name = parts[-1].lower() if parts else ""
    return bool(parts) and ".." not in parts and not any(p in NOT_PUBLIC for p in parts) and \
        "answer" not in name and "matching_key" not in name


def copy_media(keys: set[str], src: Path, dst: Path, side: int = SIDE) -> list[str]:
    """Copy ``keys`` (JPEGs scaled down to ``side``); returns the keys that were missing."""
    missing = []
    for key in sorted(keys):
        if not publishable(key):
            raise ValueError(f"refusing to publish {key}")
        s, d = src / key, dst / key
        if not s.exists():
            missing.append(key)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.suffix.lower() in (".jpg", ".jpeg"):
            from PIL import Image

            with Image.open(s) as im:
                if max(im.size) > side:
                    im = im.convert("RGB")
                    im.thumbnail((side, side))
                    im.save(d, "JPEG", quality=80, optimize=True, progressive=True)
                    continue
        shutil.copyfile(s, d)
    return missing


def write_bundle(out: Path, db_file: Path, media_dir: Path, samples: list[dict], cars: list[dict],
                 credits: list[dict], meta: dict) -> list[str]:
    """Write the folder app/showcase.py serves; returns referenced media that was missing."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    finalize_db(Path(db_file), out / DB_FILE)
    missing = copy_media(referenced_media(out / DB_FILE), Path(media_dir), out / "media")
    for s in samples:
        folder = out / "samples" / s["id"]
        folder.mkdir(parents=True)
        for i, thumb in enumerate(s["thumbs"]):
            shutil.copyfile(thumb, folder / f"thumb_{i:02d}.jpg")
    meta = {**meta, "built_at": datetime.now().replace(microsecond=0).isoformat()}
    keep = ("id", "car", "label", "names", "seconds", "result")
    (out / SAMPLES_FILE).write_text(json.dumps(
        {"meta": meta, "cars": cars, "samples": [{k: s[k] for k in keep} for s in samples]}, indent=1) + "\n")
    (out / CREDITS_FILE).write_text(json.dumps(credits, indent=1) + "\n")
    return missing


# --- The cars on the lot ---------------------------------------------------------------------


def dev_invoices(seed: int = 7, count: int = 60) -> dict:
    """The dev invoices exactly as tools/make_fixtures.py makes them (held-out ones are drawn
    to keep the random sequence, then dropped unseen)."""
    import make_fixtures as mf

    rng, pool, used, out = random.Random(seed), mf.load_vin_pool(), set(), {}
    for n in range(1, count + 1):
        layout = (n - 1) % 5 + 1
        inv = mf.make_invoice(rng, n, layout, pool, used)
        if layout in mf.DEV_LAYOUTS:
            out[inv.invoice_id] = inv
    return out


def stranger(rng: random.Random, like, invoices: dict):
    """A car like ``like`` (same make, model, color) with a VIN and plate no invoice has."""
    import make_fixtures as mf
    import seed_scale_demo as sd

    pool = mf.load_vin_pool()[(like.model.make, like.model.model)]
    year = rng.choice(sorted(pool))
    vin = sd.synth_vin(rng, rng.choice(pool[year]), {i.vin for i in invoices.values() if i.vin})
    plate = sd.unique_plate(rng, like.plate_state or "CA", {i.plate for i in invoices.values() if i.plate})
    return replace(like, invoice_id="", vin=vin, year=year, plate=plate, plate_state=like.plate_state or "CA",
                   mileage=rng.randint(20_000, 90_000))


def make_cars(folder: Path) -> tuple[list[dict], dict[str, list[Path]], dict[str, dict]]:
    """Photos of the four cars: (cars for the gallery, photo paths per car, exterior per car)."""
    import make_fixtures as mf
    import seed_scale_demo as sd

    invoices = dev_invoices()
    with open(FIXTURES / "answer_key.csv", newline="") as fh:
        key = {r["invoice_id"]: r for r in csv.DictReader(fh) if r["split"] == "dev"}
    rng = random.Random(SEED)
    ext = sd.exterior_rows()
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    cars, photos, exteriors = [], {}, {}
    for car_id, iid, shots, note in CARS:
        if iid:
            inv = invoices[iid]
            assert (inv.vin or "") == key[iid]["vin"], f"{iid}: regenerated invoice differs from the answer key"
            taken = datetime.combine(inv.invoice_date, datetime.min.time()) + timedelta(days=rng.randint(1, 4),
                                                                                        hours=rng.randint(9, 15))
        else:
            inv = stranger(rng, invoices["inv_018"], invoices)  # a silver Corolla, like many others
            taken = datetime.now().replace(second=0, microsecond=0) - timedelta(hours=2)
        row = rng.choice(ext[(inv.model.make, inv.model.model, inv.color)])
        shots_img = sd.car_photos(inv, row, rng)
        device, gps = rng.choice(mf.DEVICES), mf.shop_gps(inv.vendor)
        paths = []
        for k, kind in enumerate(shots):
            p = folder / f"{car_id}_{kind}.jpg"
            shots_img[kind].save(p, "JPEG", quality=88, exif=mf._exif_bytes(taken + timedelta(minutes=k), gps, device))
            paths.append(p)
        cars.append({"id": car_id, "title": f"{inv.color.capitalize()} {inv.model.make} {inv.model.model}",
                     "note": note})
        photos[car_id] = paths
        if "exterior" in shots:
            exteriors[car_id] = row
    return cars, photos, exteriors


def credits(exteriors: dict[str, dict]) -> list[dict]:
    """Every Commons photo in the bundle: the demo's photo sets and the sample cars."""
    import make_fixtures as mf

    rows = {r["photo"]: r for r in mf.load_exteriors()}
    used: dict[str, list[str]] = {}
    with open(FIXTURES / "matching_key.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source"].startswith("commons:") and r["invoice_id"] != DEMO_SAMPLE:
                used.setdefault(r["source"].removeprefix("commons:"), []).append("demo photo set " + r["set_id"])
    for car, row in exteriors.items():
        used.setdefault(row["photo"], []).append(f"sample {car}")
    fields = ("photo", "title", "author", "license", "license_url", "page_url")
    return [{**{k: rows[p].get(k, "") for k in fields}, "used_for": sorted(set(u))} for p, u in sorted(used.items())]


# --- Main --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama vision model ({DEFAULT_MODEL})")
    ap.add_argument("--no-vision", action="store_true", help="rules + OCR + barcode + fast-alpr only")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    import os

    import seed_demo

    from app.extract.vision import VisionModel
    from app.models import make_sessionmaker
    from app.settings import get_settings

    if not (FIXTURES / "matching_key.csv").exists():
        print("no matching scenario: run python tools/make_fixtures.py --difficulty hard (then --matching-only)")
        return 1
    model = None if args.no_vision else args.model
    BUILD.mkdir(parents=True, exist_ok=True)
    for name in ("extract_cache", "ocr_cache"):  # the regular caches: known files need no OCR or model
        link = BUILD / name
        if not link.is_symlink():
            shutil.rmtree(link, ignore_errors=True)
            link.symlink_to(ROOT / "data" / name, target_is_directory=True)
    print("1/4 seeding a fresh demo database from the caches...")
    rc = seed_demo.main(["--data", str(BUILD)] + (["--model", model] if model else ["--no-vision"]))
    if rc:
        return rc

    print("\n2/4 drawing photos of the cars on the lot...")
    cars, photos, exteriors = make_cars(BUILD / "lot_photos")

    print("\n3/4 looking them up with the full pipeline...")
    settings = get_settings(data_dir=BUILD, database_url=f"sqlite:///{BUILD / DB_FILE}")
    sessions = make_sessionmaker(settings.database_url)
    host = os.getenv("OLLAMA_HOST", "").strip() or "http://localhost:11434"
    vision = (lambda: VisionModel(model, host=host)) if model else (lambda: None)
    read_photos = lookup.default_reader(settings, vision)
    samples = []
    for car in cars:
        paths = photos[car["id"]]
        runs = [(f"{car['id']}-{p.stem.split('_', 1)[1].lower()}", KINDS[p.stem.split("_", 1)[1]], [p]) for p in paths]
        runs.append((f"{car['id']}-all", "All photos", paths))
        for sid, label, ps in runs:
            s = run_sample(sessions, read_photos, sid, car["id"], label, ps, model is not None, BUILD / "lookups")
            r = s["result"]
            found = ", ".join(x["invoice_id"] for x in r["items"][:3]) or "-"
            print(f"  {sid:16} {r['kind']:9} {found:28} {s['seconds']:5.1f} s")
            samples.append(s)
    sessions.kw["bind"].dispose()

    print("\n4/4 writing the bundle...")
    meta = {"vision_model": model, "pipeline": "rules + OCR + barcode + plate reader" + (" + AI" if model else ""),
            "invoices": None}
    con = sqlite3.connect(BUILD / DB_FILE)
    meta["invoices"] = con.execute("SELECT count(*) FROM invoices").fetchone()[0]
    con.close()
    missing = write_bundle(Path(args.out), BUILD / DB_FILE, settings.media_dir, samples, cars,
                           credits(exteriors), meta)
    if missing:
        print(f"  missing media (not bundled): {missing}")
    size = sum(p.stat().st_size for p in Path(args.out).rglob("*") if p.is_file())
    print(f"showcase bundle: {args.out} ({size / 1e6:.1f} MB, {len(samples)} saved lookups)")
    print("serve it: DEMO_MODE=showcase uvicorn app.main:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
