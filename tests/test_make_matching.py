"""The generator's matching scenario: photo sets per dev vehicle, with the hard cases."""

import random
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_fixtures as mf  # noqa: E402


@pytest.fixture(scope="module")
def invoices():
    rng, pool, used = random.Random(7), mf.load_vin_pool(), set()
    out = [mf.make_invoice(rng, n, (n - 1) % 5 + 1, pool, used) for n in range(1, 61)]
    return [i for i in out if i.layout in mf.DEV_LAYOUTS]


def exteriors_for(invoices):
    """Two fake Commons photos for every make/model/color, as exteriors.csv rows."""
    rows = []
    for i in invoices:
        for n in (1, 2):
            rows.append({"photo": f"{i.model.make}_{i.model.model}_{i.color}_{n}.jpg", "make": i.model.make,
                         "model": i.model.model, "color": i.color or "white",
                         "body_style": mf.body_style(i.model)})
    return rows


@pytest.fixture(scope="module")
def plan(invoices):
    return mf.plan_matching(invoices, seed=7, exteriors=exteriors_for(invoices))


def test_every_dev_invoice_but_one_has_a_photo_set(invoices, plan):
    with_sets = {s.invoice_id for s in plan if s.invoice_id}
    missing = {i.invoice_id for i in invoices} - with_sets
    assert len(missing) == 1
    assert [s.case for s in plan].count("invoice_without_photos") == 0  # the invoice has no set at all


def test_sets_have_the_usual_shots_and_times_near_the_invoice(invoices, plan):
    by_id = {i.invoice_id: i for i in invoices}
    for s in plan:
        assert s.shots, s
        for shot, when in zip(s.shots, s.times):
            assert shot in ("exterior", "vinplate", "sticker", "odometer", "plate")
        assert s.times == sorted(s.times) and s.times[-1] - s.times[0] < timedelta(minutes=10)
        if s.invoice_id and s.case != "no_exif_batch":
            assert 0 <= (by_id[s.invoice_id].invoice_date - s.times[0].date()).days <= 3
        assert s.uploaded >= s.times[-1]
        if s.case == "normal":
            assert "exterior" in s.shots and "odometer" in s.shots and "plate" in s.shots
            assert {"vinplate", "sticker"} & set(s.shots)


def test_lookalike_same_model_color_shop_and_day_without_invoice(plan):
    [twin] = [s for s in plan if s.case == "lookalike_no_invoice"]
    [orig] = [s for s in plan if s.case == "lookalike_original"]
    assert twin.invoice_id is None and orig.invoice_id
    assert (twin.vehicle.model, twin.color) == (orig.vehicle.model, orig.color)
    assert twin.vehicle.vendor == orig.vehicle.vendor and twin.uploader == orig.uploader
    assert twin.times[0].date() == orig.times[0].date()
    gap = twin.times[0] - orig.times[-1]
    assert timedelta(minutes=10) < gap < timedelta(minutes=30)
    assert twin.vin != orig.vin and twin.plate != orig.plate
    assert twin.exterior != orig.exterior  # the other Commons photo of the same make/model/color


def test_hard_cases_present(plan):
    cases = [s.case for s in plan]
    assert cases.count("no_vin_or_plate") == 2
    for s in plan:
        if s.case == "no_vin_or_plate":
            assert set(s.shots) <= {"exterior", "odometer"}
    batch = [s for s in plan if s.case == "no_exif_batch"]
    assert len(batch) == 2 and not any(s.exif for s in batch)
    assert batch[0].uploader == batch[1].uploader
    assert abs(batch[0].uploaded - batch[1].uploaded) < timedelta(minutes=10)
    [stray] = [s for s in plan if s.case == "no_invoice"]
    assert stray.invoice_id is None
    assert 0.15 < sum(not s.exif for s in plan) / len(plan) < 0.5


def test_vins_and_plates_are_unique_per_vehicle(plan):
    assert len({s.vin for s in plan}) == len(plan)
    assert len({(s.plate, s.state) for s in plan}) == len(plan)


def test_plan_is_deterministic(invoices, plan):
    again = mf.plan_matching(invoices, seed=7, exteriors=exteriors_for(invoices))
    assert [(s.set_id, s.vin, s.times, s.shots) for s in again] == [(s.set_id, s.vin, s.times, s.shots) for s in plan]


def test_without_exteriors_sets_have_no_exterior_shot(invoices):
    plan = mf.plan_matching(invoices, seed=7, exteriors=[])
    assert not any("exterior" in s.shots for s in plan)
