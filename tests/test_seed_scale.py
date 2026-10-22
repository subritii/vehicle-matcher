"""The scale demo generator: realistic invoice history, demo cars, photos, snapshot restore.
No lookups are run (the demo photos are only written, never read back by the engines)."""

import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_fixtures as mf  # noqa: E402
import reset_scale_demo as rs  # noqa: E402
import seed_scale_demo as sd  # noqa: E402

from app.extract.vin import is_valid  # noqa: E402

TODAY = date(2026, 10, 13)
ALL_EXTERIORS = {c for c in sd.POPULAR}  # every popular combination has a Commons exterior


@pytest.fixture(scope="module")
def plan():
    return sd.plan(300, seed=7, today=TODAY, exteriors=ALL_EXTERIORS)


def test_vins_keep_real_manufacturer_codes_and_a_valid_check_digit(plan):
    pool = mf.load_vin_pool()
    for inv in plan.invoices:
        assert len(inv.vin) == 17 and is_valid(inv.vin), inv.vin
        bases = pool[(inv.model.make, inv.model.model)][inv.year]
        assert any(inv.vin[:8] == b[:8] and inv.vin[9:11] == b[9:11] for b in bases)
    vins = [i.vin for i in plan.invoices] + [plan.stranger.vin]
    assert len(set(vins)) == len(vins)


def test_every_field_is_printed_and_plates_are_unique_without_o_i_q(plan):
    plates = [i.plate for i in plan.invoices] + [plan.stranger.plate]
    assert len(set(plates)) == len(plates)
    for inv in plan.invoices:
        assert inv.vin and inv.plate and inv.plate_state and inv.color and inv.color_printed and inv.mileage
        assert not set(inv.plate) & set("OIQ") and inv.layout in mf.DEV_LAYOUTS


def test_dates_cover_twelve_months_and_look_alikes_are_common(plan):
    days = [(TODAY - i.invoice_date).days for i in plan.invoices]
    assert min(days) >= 0 and max(days) <= 364 and max(days) - min(days) > 300
    combos = Counter((i.model.make, i.model.model, i.color) for i in plan.invoices)
    assert sum(n for n in combos.values() if n >= 10) > len(plan.invoices) / 2  # most cars have many look-alikes


def test_demo_cars_are_recent_look_alikes_with_an_exterior_and_the_stranger_has_no_invoice(plan):
    assert len(plan.demo) == sd.DEMO_CARS and all(d in plan.invoices for d in plan.demo)
    combos = Counter((i.model, i.color) for i in plan.invoices)
    for d in plan.demo:
        assert 1 <= (TODAY - d.invoice_date).days <= 6
        assert (d.model.make, d.model.model, d.color) in ALL_EXTERIORS and combos[(d.model, d.color)] >= 3
    s = plan.stranger
    assert s.vin not in {i.vin for i in plan.invoices} and s.plate not in {i.plate for i in plan.invoices}
    assert (s.model.make, s.model.model, s.color) in ALL_EXTERIORS


def test_plan_is_deterministic():
    a = sd.plan(60, seed=3, today=TODAY, exteriors=ALL_EXTERIORS)
    b = sd.plan(60, seed=3, today=TODAY, exteriors=ALL_EXTERIORS)
    assert [(i.vin, i.plate, i.invoice_date) for i in a.invoices] == [(i.vin, i.plate, i.invoice_date) for i in b.invoices]


def test_typed_pdfs_are_read_completely_without_ocr(tmp_path, plan):
    from app.extract.invoice import FIELD_NAMES, extract_invoice

    sample = plan.invoices[:40] + [i for i in plan.invoices if i.layout == 2][:15]
    assert {i.layout for i in sample} == set(mf.DEV_LAYOUTS)
    for inv in sample:
        f = extract_invoice(sd.render_pdf(inv, tmp_path), ocr=sd.NoOcr(), vpic=False)
        assert f.text_source == "pdf_text" and not f.other_reasons, inv.invoice_id  # OCR never needed
        assert set(FIELD_NAMES) <= set(f.fields), (inv.invoice_id, set(FIELD_NAMES) - set(f.fields))
        assert (f.value("vin"), f.value("plate"), f.value("mileage"), f.value("color")) == (
            inv.vin, inv.plate, inv.mileage, inv.color), inv.invoice_id


def test_history_row_matches_what_the_pipeline_would_store(tmp_path, plan):
    from app.models import index_keys
    from app.storage import LocalStorage

    inv = plan.invoices[1]
    row = index_keys(sd.history_row(inv, LocalStorage(tmp_path / "media"), sd.render_pdf(inv, tmp_path), "scale"))
    assert (row.vin, row.plate_key, row.color, row.mileage, row.date) == (
        inv.vin, inv.plate, inv.color, inv.mileage, inv.invoice_date)
    assert row.obs_json["vin_trusted"] and row.extracted_json["fields"]["make"]["value"] == inv.model.make
    assert (tmp_path / "media" / row.image_path).exists() and (tmp_path / "media" / row.file_path).exists()


def test_demo_photos_have_neutral_names_and_an_answer_key(tmp_path, plan, monkeypatch):
    ext = {c: [{"photo": "x.jpg", "make": c[0], "model": c[1], "color": c[2], "title": "A car", "author": "Someone",
                "license": "CC BY-SA 4.0", "page_url": "https://commons.example/x"}] for c in ALL_EXTERIORS}
    monkeypatch.setattr(sd, "exterior_rows", lambda: ext)
    monkeypatch.setattr(mf, "_exterior_image", lambda row, rng: mf.Image.new("RGB", (640, 480), (200, 200, 200)))
    rows = sd.write_demo_photos(plan, tmp_path / "demo_photos", seed=7, now=datetime(2026, 10, 13, 15, 0),
                                invoice_count=len(plan.invoices))
    names = sorted(p.name for p in (tmp_path / "demo_photos").iterdir())
    assert names == ["ANSWERS.md"] + sorted(f"car{k}_{kind}.jpg" for k in range(1, 7)
                                            for kind in ("VIN", "plate", "odometer", "exterior"))
    answers = (tmp_path / "demo_photos" / "ANSWERS.md").read_text()
    assert sum(r["invoice_id"] is None for r in rows) == 1 and "this car has no invoice" in answers
    for r in rows:
        if r["invoice_id"]:
            assert f"**{r['invoice_id']}**" in answers and r["inv"].vin in answers
    exif = mf.Image.open(tmp_path / "demo_photos" / "car1_VIN.jpg").getexif()
    assert exif.get_ifd(0x8769)[0x9003].startswith("2026:10:13")  # taken today, on the lot


def test_demo_photos_folder_is_emptied_not_replaced(tmp_path, plan, monkeypatch):
    """A Docker bind mount can't be removed and recreated: only its contents are replaced."""
    ext = {c: [{"photo": "x.jpg", "make": c[0], "model": c[1], "color": c[2]}] for c in ALL_EXTERIORS}
    monkeypatch.setattr(sd, "exterior_rows", lambda: ext)
    monkeypatch.setattr(mf, "_exterior_image", lambda row, rng: mf.Image.new("RGB", (64, 48), (200, 200, 200)))
    out = tmp_path / "demo_photos"
    (out / "old").mkdir(parents=True)
    (out / "car9_VIN.jpg").write_bytes(b"old")
    removed = []
    monkeypatch.setattr(sd.shutil, "rmtree", lambda p, *a, **k: removed.append(Path(p)) or None)
    sd.write_demo_photos(plan, out, seed=7, now=datetime(2026, 10, 13, 15, 0), invoice_count=len(plan.invoices))
    assert out not in removed and removed == [out / "old"]
    assert not (out / "car9_VIN.jpg").exists() and (out / "car1_VIN.jpg").exists()


def test_reset_restores_the_snapshot_in_place(tmp_path):
    from app.models import Invoice, make_sessionmaker

    data = tmp_path / "scale"
    data.mkdir()
    s = make_sessionmaker(f"sqlite:///{data / 'app.db'}")
    with s() as db:
        db.add(Invoice(id="keep", file_path="x.pdf", obs_json={}, extracted_json={}))
        db.commit()
    sd.snapshot(data / "app.db", data / "snapshot.db")
    with s() as db:  # a demo take: something saved, then a stray lookup folder
        db.add(Invoice(id="new", file_path="y.pdf", obs_json={}, extracted_json={}))
        db.commit()
    (data / "lookups" / "abc").mkdir(parents=True)
    (data / "media" / "found" / "x").mkdir(parents=True)
    rs.restore(data)
    with s() as db:  # the running app's connections see the snapshot
        assert [i.id for i in db.query(Invoice)] == ["keep"]
    assert not (data / "lookups" / "abc").exists() and not (data / "media" / "found").exists()
