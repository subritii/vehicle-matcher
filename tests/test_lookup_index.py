"""Find invoice by photo at scale: indexed candidate search, "many fit", and the explanation."""

import random
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text

from app import lookup
from app.match.cluster import Cluster as MC
from app.match.cluster import PhotoObs
from app.models import Invoice, index_keys, make_sessionmaker
from app.pipeline import invoice_obs_to_json
from app.match.score import InvoiceObs

HONDA = "1HGCM82633A004352"
TODAY = datetime(2026, 5, 6, 9, 0)


def row(iid, **obs) -> Invoice:
    o = InvoiceObs(iid, **obs)
    inv = Invoice(id=iid, file_path=f"x/{iid}.pdf", obs_json=invoice_obs_to_json(o), extracted_json={"fields": {}},
                  date=o.date, mileage=o.mileage)
    index_keys(inv)
    return inv


def sessions(tmp_path, rows):
    s = make_sessionmaker(f"sqlite:///{tmp_path / 'i.db'}")
    with s() as db:
        db.add_all(rows)
        db.commit()
    return s


def vin_photo(vin=HONDA, trusted=True):
    return PhotoObs("1", time=TODAY, time_from_exif=True, shot_type="door_jamb", vin=vin, vin_trusted=trusted,
                    vin_source="barcode" if trusted else "ocr:p")


def plate_photo(plate="7KJD221", state="CA", trusted=True):
    return PhotoObs("2", time=TODAY, time_from_exif=True, shot_type="plate", plate=plate, plate_state=state,
                    plate_source="alpr", plate_trusted=trusted)


def exterior(color="white"):
    return PhotoObs("3", time=TODAY, time_from_exif=True, shot_type="exterior", color=color, body_style="sedan")


def odometer(miles=60513):
    return PhotoObs("4", time=TODAY, time_from_exif=True, shot_type="odometer", odometer=miles, odometer_source="ocr:p")


# --- Index keys and migration ------------------------------------------------------------


def test_index_keys_come_from_the_invoice_observation():
    inv = row("i", vin=HONDA, plate="0DT-R801", plate_state="CA", make="Honda", model="Accord", color="Silver")
    assert (inv.vin, inv.vin_p1, inv.vin_p2, inv.vin_p3, inv.vin_p4) == (HONDA, "1HGCM", "8263", "3A00", "4352")
    assert inv.plate_key == "0DTR801" and inv.make_key == "honda" and inv.model_key == "accord"
    assert inv.color == "silver"


def test_old_database_is_migrated_and_backfilled(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = create_engine(url)
    with eng.begin() as c:  # the invoices table as it was before the lookup columns
        c.execute(text("""CREATE TABLE invoices (id VARCHAR(40) PRIMARY KEY, vehicle_id INTEGER, vendor VARCHAR(120),
            invoice_no VARCHAR(40), date DATE, mileage INTEGER, total VARCHAR(20), file_path VARCHAR(300),
            image_path VARCHAR(300), file_kind VARCHAR(10), extracted_json JSON, obs_json JSON, status VARCHAR(12),
            created_at DATETIME)"""))
        c.execute(text("INSERT INTO invoices (id, file_path, file_kind, extracted_json, obs_json, status) "
                       "VALUES ('old1', 'x.pdf', 'pdf', '{}', :o, 'unlinked')"),
                  {"o": '{"id": "old1", "vin": "%s", "plate": "7KJD221", "color": "red", "make": "Honda"}' % HONDA})
    s = make_sessionmaker(url)
    with s() as db:
        inv = db.get(Invoice, "old1")
        assert inv.vin == HONDA and inv.plate_key == "7KJD221" and inv.color == "red" and inv.make_key == "honda"
    with eng.connect() as c:
        names = {r[1] for r in c.execute(text("PRAGMA index_list('invoices')"))}
    assert {"ix_invoices_vin", "ix_invoices_plate_key", "ix_invoices_vin_p1"} <= names


# --- Candidate search --------------------------------------------------------------------


def three_off(vin):
    out = list(vin)
    for i in (2, 9, 15):
        out[i] = "X" if out[i] != "X" else "Y"
    return "".join(out)


def test_vin_exact_and_misread_vins_are_found_by_index(tmp_path):
    s = sessions(tmp_path, [row("exact", vin=HONDA, vin_trusted=True), row("near", vin=three_off(HONDA), color="white"),
                            row("other", vin="5YJ3E1EA4JF451621", vin_trusted=True)])
    with s() as db:
        found = lookup.candidate_rows(db, MC("x", [vin_photo()]))
    assert {r.id for r in found.rows} == {"exact", "near"}
    assert any("1 exact match" in x for x in found.searches) and any("a few characters off" in x for x in found.searches)


def test_plate_exact_folded_cut_short_and_one_off(tmp_path):
    s = sessions(tmp_path, [row("same", plate="7KJD221", plate_state="CA"), row("ofold", plate="7KJD22I"),
                            row("short", plate="7KJD"), row("one", plate="7KJD227"), row("far", plate="8BWD418")])
    with s() as db:
        found = lookup.candidate_rows(db, MC("x", [plate_photo()]))
    assert {r.id for r in found.rows} == {"same", "ofold", "short", "one"}


def test_weak_signals_only_on_a_short_list(tmp_path):
    rows = [row("odo", mileage=60700), row("color_date", color="white", date=TODAY.date() - timedelta(days=2)),
            row("color_old", color="white", date=TODAY.date() - timedelta(days=200)),
            row("nothing", color="red", mileage=10)]
    s = sessions(tmp_path, rows)
    with s() as db:
        found = lookup.candidate_rows(db, MC("x", [exterior("white"), odometer()]))
    assert {r.id for r in found.rows} == {"odo", "color_date"}


def random_invoice(rng, n):
    makes = [("Honda", "Accord"), ("Toyota", "Camry"), ("Ford", "Escape")]
    make, model = rng.choice(makes)
    vin = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17))
    return row(f"r{n:03d}", vin=vin, vin_trusted=rng.random() < 0.8, plate=f"{rng.randint(1000000, 9999999)}",
               plate_state="CA", make=make, model=model, color=rng.choice(["white", "silver", "red"]),
               mileage=rng.randint(1000, 120000), date=TODAY.date() - timedelta(days=rng.randint(0, 365)))


def test_indexed_search_gives_the_same_answer_as_scoring_every_invoice(tmp_path):
    rng = random.Random(5)
    invoices = [random_invoice(rng, n) for n in range(400)]
    s = sessions(tmp_path, invoices)
    with s() as db:
        all_rows = db.query(Invoice).all()
        for k in range(60):
            target = rng.choice(all_rows).obs_json
            photos = []
            if rng.random() < 0.5:
                photos.append(vin_photo(target["vin"] if rng.random() < 0.7 else three_off(target["vin"]),
                                        trusted=rng.random() < 0.7))
            if rng.random() < 0.5:
                photos.append(plate_photo(target["plate"], "CA", trusted=rng.random() < 0.5))
            if rng.random() < 0.6:
                photos.append(exterior(target["color"] if rng.random() < 0.8 else "black"))
            if rng.random() < 0.5:
                photos.append(odometer(target["mileage"] + rng.randint(-300, 300)))
            if not photos:
                continue
            c = MC("x", photos)
            fast = lookup.rank(c, lookup.candidate_rows(db, c).rows)
            slow = lookup.rank(c, all_rows)
            assert fast == slow, k


# --- Result kinds and the explanation --------------------------------------------------------


def test_exterior_alone_fitting_many_invoices_does_not_pick_one(tmp_path):
    rows = [row(f"w{n}", color="white", date=TODAY.date() - timedelta(days=n)) for n in range(1, 6)]
    s = sessions(tmp_path, rows + [row("red", color="red")])
    with s() as db:
        res = lookup.find_invoices([exterior("white")], db)
    assert res["kind"] == "many" and res["fits"]["count"] == 5 and res["fits"]["what"] == "white car"
    assert [x["invoice_id"] for x in res["items"]] == ["w1", "w2", "w3", "w4", "w5"]  # closest dates first
    assert "VIN" in res["why"] and "plate" in res["why"]


def test_search_lines_keep_make_capitalized(tmp_path):
    from app.extract.vin import VpicDecode

    s = sessions(tmp_path, [row("j", make="Jeep", color="silver", date=TODAY.date())])
    vpic = lambda v: VpicDecode(vin=v, ok=True, make="JEEP", model="Grand Cherokee", year=2014, error_codes=("0",))  # noqa: E731
    with s() as db:
        found = lookup.candidate_rows(db, MC("x", [vin_photo("1C4RJFAG0EC808809"), exterior("silver")]), vpic)
    assert "Silver or Jeep, dated within a week of the photos → 1 invoice" in found.searches


def test_one_fitting_invoice_is_a_possible_match(tmp_path):
    s = sessions(tmp_path, [row("only", color="white", date=TODAY.date() - timedelta(days=1)), row("red", color="red")])
    with s() as db:
        res = lookup.find_invoices([exterior("white")], db)
    assert res["kind"] == "possible" and [x["invoice_id"] for x in res["items"]] == ["only"]


def test_confident_result_explains_the_search_and_the_decision(tmp_path):
    s = sessions(tmp_path, [row("inv1", vin=HONDA, vin_trusted=True, color="white"), row("inv2", color="white")])
    with s() as db:
        res = lookup.find_invoices([vin_photo(), exterior("white")], db)
    assert res["kind"] == "confident" and res["items"][0]["invoice_id"] == "inv1"
    assert f"Looked up VIN {HONDA} among 2 invoices → 1 exact match" in res["searches"]
    assert "confirmed" in res["why"]
    assert res["items"][0]["agree"] and "VIN" in res["items"][0]["agree"][0]


def test_nothing_read_is_no_match(tmp_path):
    s = sessions(tmp_path, [row("inv1", color="white")])
    with s() as db:
        res = lookup.find_invoices([PhotoObs("9", shot_type="other")], db)
    assert res["kind"] == "none" and "Nothing that identifies" in res["why"]


def test_photo_reads_say_which_reader_produced_each_value():
    from app.extract.invoice import FieldValue
    from app.extract.photo import PhotoFields

    pf = PhotoFields("a.jpg", fields={"shot_type": FieldValue("door_jamb", 0.5, "", "vision:m"),
                                      "vin": FieldValue(HONDA, 0.99, "", "barcode"),
                                      "plate": FieldValue("7KJD221", 0.9, "", "alpr"),
                                      "odometer": FieldValue(60513, 0.85, "", "ocr:paddle"),
                                      "color": FieldValue("white", 0.5, "", "vision:m")})
    [p] = lookup.photo_reads([pf], ["IMG_1.jpg"])
    assert p["name"] == "IMG_1.jpg" and p["type"] == "Door sticker" and p["type_by"] == "AI"
    assert ("VIN", HONDA, "barcode") in p["reads"] and ("Plate", "7KJD221", "plate reader") in p["reads"]
    assert ("Odometer", "60,513 miles", "OCR") in p["reads"] and ("Color", "White", "AI") in p["reads"]
