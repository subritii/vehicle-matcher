"""Build tools/vin_pool.json: synthetic VINs that NHTSA vPIC decodes cleanly.

For each catalog model and model year in tools/make_fixtures.py, this makes candidate
VINs (real WMI, a VDS pattern for the model, year code, plant code, random serial,
correct check digit), decodes them with vPIC, and keeps only those that decode with
error code exactly "0" to the intended make, model and model year.

If a model's VDS pattern doesn't decode cleanly for a year, vPIC's ``PossibleValues``
hint (allowed characters per bad position) is used to correct it, one position at a
time, preferring choices that keep the intended make and model.

The pool is committed, so make_fixtures.py stays deterministic and offline. Re-run
this only when the catalog changes (needs network):

    python tools/build_vin_pool.py [--per-year 4] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fixtures import CATALOG, VIN_POOL_PATH, YEAR_CODES, Model, with_check_digit  # noqa: E402

VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles"
VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"

# Extra starting patterns (VDS + plant) for model generations the catalog's single
# pattern doesn't cover.
EXTRA_STARTS = {
    ("Chevrolet", "Silverado 1500"): ("UKRECG", "UYDEDG"),
    ("Nissan", "Altima"): ("AL3APN",),
}


def _client() -> httpx.Client:
    try:  # python.org builds may lack system CA certs
        import certifi
        return httpx.Client(timeout=60, verify=certifi.where())
    except ImportError:
        return httpx.Client(timeout=60)


def decode_batch(client: httpx.Client, vins: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(vins), 50):  # vPIC batch limit
        chunk = vins[i:i + 50]
        r = client.post(f"{VPIC}/DecodeVINValuesBatch/", data={"format": "json", "data": ";".join(chunk)})
        r.raise_for_status()
        out.extend(r.json()["Results"])
        time.sleep(0.5)
    return out


def squash(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def model_matches(row: dict, model: Model) -> bool:
    # vPIC splits some names: "Silverado 1500" is Model "Silverado" + Series "1500".
    want = squash(model.model)
    return want in (squash(row["Model"]), squash(row["Model"]) + squash(row.get("Series")))


def matches(row: dict, model: Model, year: int) -> bool:
    return (row["ErrorCode"] == "0" and squash(row["Make"]) == squash(model.make)
            and model_matches(row, model) and row["ModelYear"] == str(year))


def _possible(row: dict) -> list[tuple[int, str]]:
    """'(4:ABC)(7:0)' -> [(3, 'ABC'), (6, '0')] with 0-based positions."""
    return [(int(p) - 1, chars) for p, chars in re.findall(r"\((\d+):([^)]*)\)", row.get("PossibleValues") or "")]


def _score(row: dict, model: Model, year: int) -> tuple:
    return (matches(row, model, year),
            squash(row["Make"]) == squash(model.make),
            row["Model"] == "" or model_matches(row, model),
            -len(_possible(row)),
            -len(row["ErrorCode"].split(",")))


def _vin(prefix8: str, year: int, plant: str, serial: str) -> str:
    return with_check_digit(prefix8 + "0" + YEAR_CODES[year] + plant + serial)


def find_pattern(client, model: Model, year: int, starts: list[str], max_steps: int = 6) -> tuple[str, str] | None:
    """A (positions 1-8, plant) pair whose VINs decode to model/year, or None."""
    serial = "100001"
    for start in starts:
        prefix, plant = start[:8], start[8]
        for _ in range(max_steps):
            row = decode_batch(client, [_vin(prefix, year, plant, serial)])[0]
            if matches(row, model, year):
                return prefix, plant
            hints = [(p, c) for p, c in _possible(row) if p not in (8, 9) and p >= 3]
            if not hints:
                break
            pos, allowed = hints[0]
            options = []
            for ch in allowed:
                if ch not in VIN_CHARS:
                    continue
                p2, pl2 = (prefix[:pos] + ch + prefix[pos + 1:], plant) if pos < 8 else (prefix, ch)
                options.append((p2, pl2))
            if not options:
                break
            rows = decode_batch(client, [_vin(p, year, pl, serial) for p, pl in options])
            best = max(range(len(options)), key=lambda i: _score(rows[i], model, year))
            prefix, plant = options[best]
            if matches(rows[best], model, year):
                return prefix, plant
    return None


def build(per_year: int, seed: int) -> dict:
    rng = random.Random(seed)
    client = _client()
    pool: dict[str, dict[str, list[str]]] = {}
    for model in CATALOG:
        key = f"{model.make}|{model.model}"
        pool[key] = {}
        found: list[str] = []  # patterns that worked for other years of this model
        for year in range(model.years[0], model.years[1] + 1):
            base = [model.wmi + model.vds + model.plant]
            base += [model.wmi + x for x in EXTRA_STARTS.get((model.make, model.model), ())]
            starts = list(dict.fromkeys(found + base))
            pattern = find_pattern(client, model, year, starts)
            if pattern is None:
                print(f"  {key} {year}: no clean pattern, skipped")
                continue
            prefix, plant = pattern
            if prefix + plant not in found:
                found.insert(0, prefix + plant)
            candidates = sorted({_vin(prefix, year, plant, f"{rng.randint(100000, 999999)}")
                                 for _ in range(per_year * 2)})
            rows = decode_batch(client, candidates)
            keep = [v for v, row in zip(candidates, rows) if matches(row, model, year)][:per_year]
            pool[key][str(year)] = keep
            print(f"  {key} {year}: {prefix[3:]}.{plant} -> {len(keep)}/{len(candidates)} clean")
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-year", type=int, default=4, help="VINs to keep per model and year")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=VIN_POOL_PATH)
    args = ap.parse_args(argv)
    pool = build(args.per_year, args.seed)
    doc = {
        "about": "Synthetic VINs that vPIC decodes with ErrorCode 0 to the given make, model "
                 "and year. Built by tools/build_vin_pool.py; used by tools/make_fixtures.py.",
        "pool": pool,
    }
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    n = sum(len(v) for years in pool.values() for v in years.values())
    print(f"wrote {n} VINs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
