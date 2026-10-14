"""Accuracy of app/extract/photo.py per photo type and field.

Answer keys (all that exist are used, or pass --key):
  tests/fixtures/synthetic_hard/photos_answer_key.csv   generated stickers, VIN plates,
      plates, odometers for the dev vehicles, plus invoice photos as "other"
      (python tools/make_fixtures.py --difficulty hard)
  tests/fixtures/exteriors/exteriors.csv                 Wikimedia Commons exteriors
      (python tools/fetch_exteriors.py)
  tests/fixtures/private/photos/answer_key.csv           your own photos (gitignored),
      e.g. car_1/IMG_0001.jpg; same columns as the synthetic key, blanks allowed

For each true shot type: how often the shot type is right, and for each field the key
has, correct / expected. Also trusted VINs and plates, and how many of those are wrong
(target 0).

Usage:
    python tools/photo_accuracy.py [--key PATH ...] [--model NAME | --no-vision] [--no-vpic] [-v]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extract import photo as ph  # noqa: E402
from app.extract.vision import VisionModel  # noqa: E402
from app.extract.vision import from_env as vision_from_env  # noqa: E402

DEFAULT_KEYS = (
    ROOT / "tests" / "fixtures" / "synthetic_hard" / "photos_answer_key.csv",
    ROOT / "tests" / "fixtures" / "exteriors" / "exteriors.csv",
    ROOT / "tests" / "fixtures" / "private" / "photos" / "answer_key.csv",
)
FIELDS = ("vin", "plate", "plate_state", "odometer", "color", "body_style", "captured_at", "gps", "device")


def squash(s: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def expected(row: dict, name: str):
    if name == "gps":
        return (float(row["gps_lat"]), float(row["gps_lon"])) if row.get("gps_lat") and row.get("gps_lon") else None
    v = (row.get(name) or "").strip()
    return v or None


def same(name: str, want, got) -> bool:
    if got is None:
        return False
    if name in ("vin", "plate"):
        return squash(want) == squash(got)
    if name == "odometer":
        return int(want) == int(got)
    if name == "captured_at":
        return datetime.fromisoformat(want) == got
    if name == "gps":
        return abs(want[0] - got[0]) < 1e-3 and abs(want[1] - got[1]) < 1e-3
    return str(want).strip().lower() == str(got).strip().lower()


def load_rows(keys: list[Path]) -> list[dict]:
    rows = []
    for key in keys:
        with open(key, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("split", "dev") not in ("", "dev"):
                    continue  # never score holdout photos here
                r["_path"] = key.parent / r["photo"]
                r["_key"] = key.parent.name
                rows.append(r)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", type=Path, action="append", help="answer key CSV (repeatable)")
    ap.add_argument("--model", help="Ollama vision model (default: OLLAMA_MODEL)")
    ap.add_argument("--no-vision", action="store_true", help="rules only: shot type from barcode/OCR/ALPR")
    ap.add_argument("--no-vpic", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every mistake")
    args = ap.parse_args(argv)

    keys = args.key or [k for k in DEFAULT_KEYS if k.exists()]
    rows = [r for r in load_rows(keys) if r["_path"].exists()]
    if not rows:
        print("no photos found; run tools/make_fixtures.py --difficulty hard")
        return 1
    vision = None
    if not args.no_vision:
        vision = VisionModel(args.model) if args.model else vision_from_env()
        if vision is None:
            print("no vision model: pass --model, set OLLAMA_MODEL, or use --no-vision")
            return 1
    engines = ph.Engines(vision=vision)
    uploaded = {r["_path"]: datetime.fromisoformat(r["uploaded_at"]) for r in rows if r.get("uploaded_at")}

    t0 = time.perf_counter()
    results = ph.extract_photos([r["_path"] for r in rows], engines, uploaded_at=uploaded,
                                vpic=False if args.no_vpic else None)
    seconds = time.perf_counter() - t0

    shot_right = Counter()
    count = Counter()
    confusion = defaultdict(Counter)
    right = defaultdict(int)  # (shot, field)
    want_n = defaultdict(int)
    by_source = defaultdict(lambda: [0, 0])  # (field, source kind) -> [right, wrong]
    trusted = Counter()
    wrong_trusted = Counter()
    review = Counter()
    mistakes = []
    for row, res in zip(rows, results):
        shot = row["shot_type"]
        got_shot = res.value("shot_type")
        count[shot] += 1
        confusion[shot][got_shot] += 1
        shot_right[shot] += got_shot == shot
        review[shot] += res.needs_review
        if got_shot != shot:
            mistakes.append((row["photo"], "shot_type", shot, got_shot))
        for name in FIELDS:
            want = expected(row, name)
            if want is None:
                continue
            got = res.value(name)
            ok = same(name, want, got)
            want_n[(shot, name)] += 1
            right[(shot, name)] += ok
            if got is not None:
                kind = res.fields[name].source.split(":")[0]
                by_source[(name, kind)][0 if ok else 1] += 1
            if not ok:
                mistakes.append((row["photo"], name, want, got, res.fields[name].source if got is not None else ""))
        for name, ok_trust in (("vin", res.vin_trusted), ("plate", res.plate_trusted)):
            if ok_trust:
                trusted[name] += 1
                want = expected(row, name)
                if want is None or not same(name, want, res.value(name)):
                    wrong_trusted[name] += 1
                    mistakes.append((row["photo"], f"WRONG TRUSTED {name.upper()}", want, res.value(name)))

    n = len(rows)
    mode = f"vision model {vision.model}" if vision else "no vision model (rules)"
    print(f"\n{n} photos from {', '.join(sorted({r['_key'] for r in rows}))}; {mode}\n")
    shots = [s for s in ph.SHOT_TYPES if count[s]]
    print(f"{'shot type':12}{'photos':>8}{'shot right':>14}  fields (right/expected)")
    print("-" * 96)
    for s in shots:
        cells = [f"{f} {right[(s, f)]}/{want_n[(s, f)]}" for f in FIELDS if want_n[(s, f)]]
        print(f"{s:12}{count[s]:>8}{f'{shot_right[s]}/{count[s]} {100 * shot_right[s] / count[s]:3.0f}%':>14}  "
              + ", ".join(cells))
    print("-" * 96)
    total_right = sum(shot_right.values())
    print(f"{'all':12}{n:>8}{f'{total_right}/{n} {100 * total_right / n:3.0f}%':>14}")

    print(f"\n{'field':14}{'right/expected':>18}")
    for name in FIELDS:
        e = sum(want_n[(s, name)] for s in shots)
        if e:
            c = sum(right[(s, name)] for s in shots)
            print(f"{name:14}{f'{c}/{e} {100 * c / e:5.1f}%':>18}")

    print("\nShot type confusion (true -> predicted):")
    for s in shots:
        print(f"  {s:12} " + ", ".join(f"{p} {k}" for p, k in confusion[s].most_common()))
    print("\nValues by source (right / wrong):")
    for (name, kind), (r_, w_) in sorted(by_source.items()):
        print(f"  {name:12} {kind:8} {r_} / {w_}")
    print(f"\nVIN trusted {trusted['vin']}, wrong + trusted {wrong_trusted['vin']}")
    print(f"Plate trusted {trusted['plate']}, wrong + trusted {wrong_trusted['plate']}")
    print("Needs review: " + ", ".join(f"{s} {review[s]}/{count[s]}" for s in shots))
    print(f"Time: {seconds:.0f} s total, {seconds / n:.1f} s/photo")
    if args.verbose:
        print("\nMistakes (photo, field, expected, got, source):")
        for m in mistakes:
            print("  ", *m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
