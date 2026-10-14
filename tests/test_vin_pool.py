"""tools/vin_pool.json (vPIC-verified synthetic VINs) and its use in make_fixtures.py.

Offline checks only; the vPIC decode itself happens in tools/build_vin_pool.py."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import make_fixtures as mf  # noqa: E402

from app.extract import vin as vinlib  # noqa: E402

POOL = mf.load_vin_pool()


def test_every_catalog_model_has_vins():
    for m in mf.CATALOG:
        assert POOL.get((m.make, m.model)), m


def test_pool_vins_are_valid_and_match_their_model_and_year():
    seen = set()
    by_key = {(m.make, m.model): m for m in mf.CATALOG}
    for key, years in POOL.items():
        model = by_key[key]
        for year, vins in years.items():
            assert model.years[0] <= year <= model.years[1]
            for v in vins:
                assert vinlib.is_valid(v), v
                assert v[:3] == model.wmi and v[9] == mf.YEAR_CODES[year], v
                assert v not in seen
                seen.add(v)


def test_generator_uses_only_pool_vins_without_repeats():
    rng, used = random.Random(7), set()
    pooled = {v for years in POOL.values() for vins in years.values() for v in vins}
    vins = []
    for n in range(1, 61):
        inv = mf.make_invoice(rng, n, (n - 1) % 5 + 1, POOL, used)
        if inv.vin:
            assert inv.vin in pooled
            assert inv.vin in POOL[(inv.model.make, inv.model.model)][inv.year]
            vins.append(inv.vin)
    assert len(vins) == len(set(vins))
