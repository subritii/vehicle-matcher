"""Build the dealership-sized demo: an invoice history, plus fresh photos of cars on the lot.

A separate database in data/scale/ (the regular demo in data/ is not touched):
- --count clean, typed PDF invoices (layouts 1-3, no handwriting), dates over the last 12
  months, with many look-alikes (popular make/model/color combinations). VINs keep a real
  manufacturer, model and year code from tools/vin_pool.json, with a new serial number and
  a valid check digit. No photos are attached.
- The first 50 invoices go through the real pipeline (PDF text layer, rules, VIN checks,
  NHTSA vPIC) and are timed. If 2,000 invoices would take over 15 minutes that way, the rest
  of the history is loaded directly as already-processed invoices; the demo cars' invoices
  always go through the real pipeline.
- The finished database is saved as data/scale/snapshot.db (tools/reset_scale_demo.py
  restores it in seconds).
- 5 invoices are the "cars on the lot": each gets brand-new photos (door-jamb VIN sticker,
  license plate close-up, odometer, and a Wikimedia Commons exterior of the same make,
  model and color), plus one car that has no invoice. Saved to ~/Desktop/demo_photos/ as
  car1_VIN.jpg ... car6_exterior.jpg, with the answers in ANSWERS.md.

No lookups are run on the demo photos.

Usage:
    python tools/seed_scale_demo.py [--count 200] [--seed 2026] [--mode auto|pipeline|history]
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import make_fixtures as mf  # noqa: E402
from PIL import Image, ImageEnhance, ImageOps  # noqa: E402

DATA = ROOT / "data" / "scale"
PHOTOS_OUT = Path.home() / "Desktop" / "demo_photos"
DEMO_CARS = 5
PIPELINE_SAMPLE = 50
MAX_PIPELINE_SECONDS_FOR_2000 = 15 * 60
SERIAL_DIGITS = "0123456789"
LAYOUT2_MAX = 26

# Popular combinations (all have Commons exteriors): most of the history, so look-alikes abound.
POPULAR = (
    ("Honda", "CR-V", "silver"), ("Toyota", "Corolla", "silver"), ("Ford", "Escape", "white"),
    ("Nissan", "Altima", "white"), ("Tesla", "Model 3", "white"), ("Jeep", "Grand Cherokee", "black"),
    ("Ford", "F-150", "silver"), ("Honda", "Civic", "gray"), ("Toyota", "Camry", "blue"),
    ("Chevrolet", "Equinox", "silver"),
)
POPULAR_SHARE = 0.65


# --- Invoices --------------------------------------------------------------------------------


def synth_vin(rng: random.Random, base: str, used: set[str]) -> str:
    """A VIN with ``base``'s manufacturer, model, year and plant codes (positions 1-8, 10-11),
    a new serial number (12-17) and a valid check digit (9)."""
    while True:
        vin = mf.with_check_digit(base[:11] + "".join(rng.choice(SERIAL_DIGITS) for _ in range(6)))
        if vin not in used:
            used.add(vin)
            return vin


def unique_plate(rng: random.Random, state: str, used: set[str]) -> str:
    while True:
        plate = mf.make_plate(rng, state)  # letters never include O, I or Q
        if plate not in used:
            used.add(plate)
            return plate


def pick_combo(rng: random.Random, pool) -> tuple[mf.Model, str]:
    if rng.random() < POPULAR_SHARE:
        make, model, color = rng.choice(POPULAR)
    else:
        m = rng.choice([m for m in mf.CATALOG if pool.get((m.make, m.model))])
        make, model, color = m.make, m.model, rng.choice(list(mf.COLORS))
    return next(m for m in mf.CATALOG if (m.make, m.model) == (make, model)), color


def scale_invoice(rng: random.Random, n: int, today: date, pool, used_vins: set[str], used_plates: set[str],
                  combo: tuple[mf.Model, str] | None = None, days_ago: int | None = None) -> mf.Invoice:
    """One clean, fully typed invoice: VIN, plate, color, mileage and date always printed."""
    model, color = combo or pick_combo(rng, pool)
    years = pool[(model.make, model.model)]
    year = rng.choice(sorted(years))
    trim = rng.choice(model.trims)
    # Layout 2's YEAR/MAKE/MODEL box holds about 26 characters; longer names would run into
    # the COLOR box next to it, so those cars get one of the other layouts.
    fits = [k for k in mf.DEV_LAYOUTS if k != 2 or len(f"{year} {model.make} {model.model} {trim}") <= LAYOUT2_MAX]
    layout = rng.choice(fits)
    base = mf.make_invoice(rng, n, layout, pool, set())  # shop, customer, line items
    state = rng.choice(list(mf.PLATE_FORMATS))
    when = today - timedelta(days=rng.randint(0, 364) if days_ago is None else days_ago)
    age = max(today.year - year, 0)
    return replace(base, invoice_id=f"inv_{n:04d}", model=model, year=year, trim=trim,
                   vin=synth_vin(rng, rng.choice(years[year]), used_vins), plate=unique_plate(rng, state, used_plates),
                   plate_state=state, color=color, color_printed=rng.choice(mf.COLORS[color]),
                   mileage=rng.randint(max(age * 6000, 1500), max(age * 16000, 9000)), invoice_date=when,
                   claim_no=base.claim_no and f"CLM-{when.year}-{rng.randint(1000000, 9999999)}")


@dataclass
class Plan:
    invoices: list[mf.Invoice]
    demo: list[mf.Invoice]  # the cars on the lot that have an invoice
    stranger: mf.Invoice  # the car on the lot without one (not in ``invoices``)


def plan(count: int, seed: int, today: date, exteriors: set[tuple[str, str, str]]) -> Plan:
    """``count`` invoices; DEMO_CARS of them (recent, popular look-alike combinations with a
    Commons exterior) are the cars on the lot."""
    rng = random.Random(f"scale-{seed}")
    pool = mf.load_vin_pool()
    used_vins: set[str] = set()
    used_plates: set[str] = set()
    demo_combos = [c for c in POPULAR if c in exteriors]
    rng.shuffle(demo_combos)
    slots = set(rng.sample(range(1, count + 1), DEMO_CARS))
    invoices, demo = [], []
    for n in range(1, count + 1):
        if n in slots:
            make, model, color = demo_combos[len(demo) % len(demo_combos)]
            m = next(x for x in mf.CATALOG if (x.make, x.model) == (make, model))
            inv = scale_invoice(rng, n, today, pool, used_vins, used_plates, (m, color), days_ago=rng.randint(1, 6))
            demo.append(inv)
        else:
            inv = scale_invoice(rng, n, today, pool, used_vins, used_plates)
        invoices.append(inv)
    make, model, color = demo_combos[DEMO_CARS % len(demo_combos)]
    m = next(x for x in mf.CATALOG if (x.make, x.model) == (make, model))
    stranger = scale_invoice(rng, count + 1, today, pool, used_vins, used_plates, (m, color), days_ago=2)
    return Plan(invoices, demo, replace(stranger, invoice_id="no_invoice"))


def render_pdf(inv: mf.Invoice, folder: Path) -> Path:
    path = folder / f"{inv.invoice_id}.pdf"
    mf.LAYOUTS[inv.layout](inv, hand=None).output(str(path))
    return path


class NoOcr:
    """Typed PDFs need no OCR; if a field is missing anyway, it stays missing (and is counted)."""

    name = "none"
    word_level = False

    def __call__(self, image):
        from app.extract.invoice import OcrFailed

        raise OcrFailed("not used for typed PDFs")

    def close(self) -> None:
        pass


def history_row(inv: mf.Invoice, storage, pdf: Path, prefix: str):
    """An invoice loaded as already-processed history: the fields as printed on the PDF."""
    from app import pipeline
    from app.match.score import InvoiceObs
    from app.models import Invoice

    values = {"vin": inv.vin, "plate": inv.plate, "plate_state": inv.plate_state, "year": inv.year,
              "make": inv.model.make, "model": inv.model.model, "color": inv.color, "mileage": inv.mileage,
              "invoice_date": inv.invoice_date.isoformat(), "vendor": inv.vendor, "invoice_no": inv.invoice_no,
              "total": mf.money(inv.total)}
    obs = InvoiceObs(inv.invoice_id, vin=inv.vin, vin_trusted=True, vin_source="pdf_text", plate=inv.plate,
                     plate_state=inv.plate_state, plate_trusted=True, make=inv.model.make, model=inv.model.model,
                     year=inv.year, color=inv.color, mileage=inv.mileage, date=inv.invoice_date)
    return Invoice(
        id=inv.invoice_id, vendor=inv.vendor, invoice_no=inv.invoice_no, date=inv.invoice_date, mileage=inv.mileage,
        total=mf.money(inv.total), file_kind="pdf",
        file_path=storage.put_file(f"{prefix}/invoices/{inv.invoice_id}.pdf", pdf),
        image_path=storage.put_image(f"{prefix}/invoices/{inv.invoice_id}_view.jpg", pipeline.invoice_image(pdf),
                                     pipeline.INVOICE_SIDE),
        extracted_json={"fields": {k: {"value": str(v), "confidence": 0.95, "source": "pdf_text"}
                                   for k, v in values.items()},
                        "vin_trusted": True, "plate_trusted": True, "review_reasons": [], "boxes": {},
                        "vin_confirmations": [{"source": "text_layer", "agrees": True, "detail": "digital PDF"}],
                        "history": True},
        obs_json=pipeline.invoice_obs_to_json(obs))


def snapshot(db_file: Path, snap: Path) -> None:
    src, dst = sqlite3.connect(db_file), sqlite3.connect(snap)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


# --- Photos of the cars on the lot ------------------------------------------------------------


def exterior_rows() -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = {}
    for r in mf.load_exteriors():
        out.setdefault((r["make"], r["model"], r["color"]), []).append(r)
    return out


def _lighting(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.12))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))
    return ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.08))


def car_photos(inv: mf.Invoice, ext: dict, rng: random.Random) -> dict[str, Image.Image]:
    """Fresh shots of one car: door-jamb VIN sticker, plate close-up, odometer (a few miles past
    the invoice), and a Commons exterior of the same make, model and color, re-framed."""
    sticker, _ = mf.draw_sticker(inv, inv.vin, rng, hard=False)
    plate = mf.draw_plate(inv, inv.plate, inv.plate_state, rng, hard=False)
    odo = mf.draw_odometer(replace(inv, mileage=inv.mileage + rng.randint(8, 140)), rng, hard=False)
    outside = mf._exterior_image(ext, rng)
    if rng.random() < 0.5:
        outside = ImageOps.mirror(outside)
    w, h = outside.size
    dx, dy = round(w * rng.uniform(0.02, 0.08)), round(h * rng.uniform(0.02, 0.08))
    outside = outside.crop((dx, dy, w - round(w * rng.uniform(0.0, 0.06)), h - round(h * rng.uniform(0.0, 0.06))))
    return {"VIN": _lighting(sticker, rng), "plate": _lighting(plate, rng), "odometer": _lighting(odo, rng),
            "exterior": _lighting(outside, rng)}


def write_demo_photos(p: Plan, out: Path, seed: int, now: datetime, invoice_count: int) -> list[dict]:
    """car1..car6 in shuffled order (the no-invoice car included), photos and ANSWERS.md."""
    rng = random.Random(f"lot-photos-{seed}-{now:%Y%m%d}")  # a seed no fixture uses
    ext = exterior_rows()
    cars = [(inv, inv.invoice_id) for inv in p.demo] + [(p.stranger, None)]
    rng.shuffle(cars)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for k, (inv, invoice_id) in enumerate(cars, start=1):
        options = ext[(inv.model.make, inv.model.model, inv.color)]
        ext_row = rng.choice(options)
        device = rng.choice(mf.DEVICES)
        gps = mf.shop_gps(inv.vendor)
        taken = now - timedelta(minutes=rng.randint(10, 90) + 12 * k)
        for i, (kind, img) in enumerate(car_photos(inv, ext_row, rng).items()):
            when = taken + timedelta(minutes=i, seconds=rng.randint(0, 59))
            img.save(out / f"car{k}_{kind}.jpg", "JPEG", quality=rng.randint(84, 92),
                     exif=mf._exif_bytes(when, gps, device))
        others = sum(1 for x in p.invoices if (x.model, x.color) == (inv.model, inv.color) and x is not inv)
        rows.append({"car": f"car{k}", "invoice_id": invoice_id, "inv": inv, "exterior": ext_row, "lookalikes": others})
    (out / "ANSWERS.md").write_text(answers_md(rows, invoice_count, now))
    return rows


def answers_md(rows: list[dict], invoice_count: int, now: datetime) -> str:
    lines = [
        "# Demo photos: answer key",
        "",
        f"Made {now:%Y-%m-%d %H:%M} for the scale demo database (data/scale, {invoice_count} invoices).",
        "Each car has four photos: car<N>_VIN.jpg (door-jamb sticker with a VIN barcode), car<N>_plate.jpg,",
        "car<N>_odometer.jpg and car<N>_exterior.jpg. Try them one at a time or together on the Find page.",
        "",
        "| Car | Invoice | Invoice no. | Shop | Invoice date | Car on the invoice | VIN | Plate | Look-alikes* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        inv = r["inv"]
        car = f"{inv.year} {inv.model.make} {inv.model.model} · {inv.color}"
        if r["invoice_id"]:
            lines.append(f"| {r['car']} | **{r['invoice_id']}** | {inv.invoice_no} | {inv.vendor} | "
                         f"{inv.invoice_date:%b %d, %Y} | {car} | `{inv.vin}` | `{inv.plate}` {inv.plate_state} | "
                         f"{r['lookalikes']} |")
        else:
            lines.append(f"| {r['car']} | **none: this car has no invoice** | - | - | - | {car} | `{inv.vin}` | "
                         f"`{inv.plate}` {inv.plate_state} | {r['lookalikes']} |")
    lines += [
        "",
        "\\* Other invoices for the same make, model and color.",
        "",
        "## What to expect",
        "",
        "- VIN sticker or plate photo of cars with an invoice: **Found it**, with that invoice.",
        "- Exterior or odometer alone: several invoices fit the description, so the tool asks for a VIN or",
        "  plate photo instead of picking one (or shows a possible match if only one fits).",
        "- The car without an invoice: no match on its VIN or plate.",
        "",
        "## Exterior photo credits (Wikimedia Commons, cropped and re-lit)",
        "",
    ]
    for r in rows:
        e = r["exterior"]
        lines.append(f"- {r['car']}: \"{e.get('title', e['photo'])}\" by {e.get('author') or 'unknown'}, "
                     f"{e.get('license') or 'see page'} ({e.get('page_url', '')})")
    return "\n".join(lines) + "\n"


# --- Main --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=200, help="invoices in the history (default 200)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=("auto", "pipeline", "history"), default="auto",
                    help="auto: real pipeline for all if 2,000 would take under 15 minutes, else history load")
    ap.add_argument("--data", default=str(DATA), help="data folder of the scale demo (default data/scale)")
    ap.add_argument("--photos", default=str(PHOTOS_OUT), help="where the demo photos go")
    ap.add_argument("--no-vpic", action="store_true", help="skip the NHTSA vPIC lookups (offline)")
    args = ap.parse_args(argv)
    if args.count < DEMO_CARS + PIPELINE_SAMPLE:
        ap.error(f"--count must be at least {DEMO_CARS + PIPELINE_SAMPLE}")

    from app import pipeline
    from app.extract import cache as ch
    from app.extract.invoice import extract_invoice
    from app.models import make_sessionmaker, reset_database
    from app.settings import get_settings
    from app.storage import LocalStorage

    ext = exterior_rows()
    if not ext:
        print("no Commons exteriors: run WIKIMEDIA_CONTACT=<email or URL> python tools/fetch_exteriors.py first")
        return 1
    data_dir = Path(args.data).resolve()
    settings = get_settings(data_dir=data_dir, database_url=f"sqlite:///{data_dir / 'app.db'}")  # never a .env DATABASE_URL
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().replace(microsecond=0)
    p = plan(args.count, args.seed, now.date(), set(ext))
    demo_ids = {inv.invoice_id for inv in p.demo}

    reset_database(settings.database_url)
    storage = LocalStorage(settings.media_dir)
    for prefix in ("scale", "uploads", "found"):
        storage.delete_prefix(prefix)
    shutil.rmtree(settings.data_dir / "lookups", ignore_errors=True)
    pdf_dir = settings.data_dir / "pdf"
    shutil.rmtree(pdf_dir, ignore_errors=True)
    pdf_dir.mkdir(parents=True)
    sessions = make_sessionmaker(settings.database_url)
    vpic = False if args.no_vpic else ch.cached_vpic(ch.DiskCache(settings.extract_cache))
    ocr = NoOcr()
    missing: dict[str, int] = {}

    def through_pipeline(db, inv) -> None:
        pdf = render_pdf(inv, pdf_dir)
        f = extract_invoice(pdf, ocr=ocr, vpic=vpic)
        for name in ("vin", "plate", "plate_state", "year", "make", "model", "color", "mileage", "invoice_date",
                     "vendor", "invoice_no", "total"):
            if name not in f.fields:
                missing[name] = missing.get(name, 0) + 1
        pipeline.add_invoice(db, storage, inv.invoice_id, pdf, "pdf", f, settings.ocr_cache, "scale")

    print(f"{args.count} invoices: timing the real pipeline on the first {PIPELINE_SAMPLE}...")
    order = sorted(p.invoices, key=lambda i: (i.invoice_id not in demo_ids, i.invoice_id))  # demo cars first
    with sessions() as db:
        t0 = time.perf_counter()
        for inv in order[:PIPELINE_SAMPLE]:
            through_pipeline(db, inv)
        db.commit()
        per = (time.perf_counter() - t0) / PIPELINE_SAMPLE
        est_2000 = per * 2000
        mode = args.mode if args.mode != "auto" else ("history" if est_2000 > MAX_PIPELINE_SECONDS_FOR_2000 else "pipeline")
        print(f"  {per:.2f} s per invoice -> about {est_2000 / 60:.0f} min for 2,000 "
              f"({per * args.count / 60:.1f} min for {args.count}); rest of the history: "
              f"{'real pipeline' if mode == 'pipeline' else 'loaded directly as already-processed invoices'}")
        t1 = time.perf_counter()
        for k, inv in enumerate(order[PIPELINE_SAMPLE:], start=1):
            if mode == "pipeline" or inv.invoice_id in demo_ids:
                through_pipeline(db, inv)
            else:
                db.add(history_row(inv, storage, render_pdf(inv, pdf_dir), "scale"))
            if k % 200 == 0:
                db.commit()
                print(f"  {PIPELINE_SAMPLE + k} / {args.count}")
        db.commit()
        rest = time.perf_counter() - t1
    total = time.perf_counter() - t0
    print(f"  done in {total:.0f} s ({rest / max(len(order) - PIPELINE_SAMPLE, 1):.2f} s per invoice after the first "
          f"{PIPELINE_SAMPLE})")
    if missing:
        print(f"  fields not found on the typed PDFs: {missing}")
    shutil.rmtree(pdf_dir, ignore_errors=True)  # the originals are in media/

    db_file = Path(settings.database_url.removeprefix("sqlite:///"))
    snapshot(db_file, settings.data_dir / "snapshot.db")
    rows = write_demo_photos(p, Path(args.photos).expanduser(), args.seed, now, args.count)
    print(f"\nsnapshot: {settings.data_dir / 'snapshot.db'} (restore: python tools/reset_scale_demo.py)")
    print(f"demo photos: {Path(args.photos).expanduser()} ({len(rows)} cars, answers in ANSWERS.md)")
    print(f"run: DATA_DIR={args.data} uvicorn app.main:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
