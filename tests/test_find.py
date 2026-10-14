"""Find invoice from photo: a lookup that changes nothing unless the user saves."""

import io
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import lookup
from app.extract.invoice import FieldValue, VinConfirmation
from app.extract.photo import PhotoFields
from app.main import create_app
from app.match.cluster import PhotoObs
from app.models import Cluster, Invoice, MatchEvent, Photo, Vehicle
from app.settings import Settings
from tests.test_web import invoice

HONDA = "1HGCM82633A004352"
WHEN = datetime(2026, 5, 6, 9, 0)  # two days after the invoices' date (2026-05-04)


def png(color="gray", fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, fmt)
    return buf.getvalue()


def fields(path, shot, **kw) -> PhotoFields:
    pf = PhotoFields(str(path), fields={"shot_type": FieldValue(shot, 0.9, "", "rules"),
                                        "captured_at": FieldValue(WHEN, 0.9, "", "exif")})
    if "vin" in kw:
        pf.fields["vin"] = FieldValue(kw["vin"], 0.99, kw["vin"], "barcode")
        pf.vin_confirmations.append(VinConfirmation("barcode", True))
    if "color" in kw:
        pf.fields["color"] = FieldValue(kw["color"], 0.5, kw["color"], "vision:m")
    return pf


# What the fake readers "see" per uploaded file name (stem).
SEEN = {
    "vin": dict(shot="door_jamb", vin=HONDA),
    "silver": dict(shot="exterior", color="silver"),
    "green": dict(shot="exterior", color="green"),
}


def obs_invoice(iid, **obs):
    inv = invoice(iid, vin=obs.get("vin"), color=obs.get("color"))
    inv.obs_json = {"id": iid, "vin": None, "vin_trusted": False, "date": "2026-05-04", **obs}
    return inv


@pytest.fixture
def client(tmp_path):
    def read_photos(paths, progress, uploaded_at):
        for step in ("vision", "ocr", "alpr"):
            progress(step)
        return [fields(p, **SEEN[p.stem.split("_", 1)[1]]) for p in paths], None

    app = create_app(Settings(tmp_path, f"sqlite:///{tmp_path / 't.db'}"), vision_factory=lambda: None,
                     read_photos=read_photos)
    with app.state.sessions() as db:
        db.add_all([
            obs_invoice("inv1", vin=HONDA, vin_trusted=True, color="silver", make="Honda", model="Accord"),
            obs_invoice("inv2", color="silver", make="Toyota", model="Camry"),
            obs_invoice("inv3", color="silver"),
            obs_invoice("inv4", color="silver", mileage=1000),
            obs_invoice("inv5", color="red"),
        ])
        db.commit()
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.app_ = app
    return c


def counts(client):
    with client.app_.state.sessions() as db:
        return {m.__name__: db.query(m).count() for m in (Invoice, Photo, Cluster, MatchEvent, Vehicle)} | {
            "linked": db.query(Invoice).filter_by(status="linked").count()}


def find(client, *names, ext="png", data=None):
    files = [("photos", (f"{n}.{ext}", data or png(), f"image/{ext}")) for n in names]
    r = client.post("/find", files=files, follow_redirects=False)
    assert r.status_code == 303, r.text[:300]
    client.app_.state.runner.executor.submit(lambda: None).result(timeout=10)
    return r.headers["location"]


# --- The page ----------------------------------------------------------------------------


def test_page_works_on_computer_and_phone(client):
    r = client.get("/find")
    assert r.status_code == 200
    html = r.text
    assert 'type="file"' in html and "multiple" in html
    for t in ("image/jpeg", "image/png", ".heic", "image/heic"):
        assert t in html
    assert "capture=" not in html  # a phone then offers gallery *and* camera
    assert 'id="drop-zone"' in html and "dragover" in html
    assert 'name="viewport"' in html
    assert 'href="/find"' in client.get("/").text  # linked from the overview and the navigation


def test_rejects_other_file_types(client):
    before = counts(client)
    r = client.post("/find", files=[("photos", ("a.gif", b"GIF89a", "image/gif"))])
    assert r.status_code == 400 and "JPG, PNG or HEIC" in r.text
    r = client.post("/find", files=[])
    assert r.status_code in (400, 422)
    assert counts(client) == before


def test_heic_photos_are_accepted(client):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    url = find(client, "vin", ext="heic", data=png(fmt="HEIF"))
    page = client.get(url)
    assert "VIN matches" in page.text
    assert client.get(f"{url}/thumb/0.jpg").headers["content-type"] == "image/jpeg"


# --- Results ------------------------------------------------------------------------------


def test_trusted_vin_gives_one_confident_result(client):
    before = counts(client)
    url = find(client, "vin", "silver")
    r = client.get(url)
    assert "VIN matches" in r.text and "inv1" in r.text and "Possible matches" not in r.text
    assert 'data-highlight="vin"' in r.text
    assert counts(client) == before  # a lookup creates and changes nothing


def test_exterior_alone_fitting_several_invoices_picks_none(client):
    url = find(client, "silver")
    r = client.get(url)
    assert "4 invoices fit this description" in r.text and "Add a photo of the VIN sticker or the license plate" in r.text
    assert 'data-fits="4"' in r.text and 'data-candidate="' not in r.text and "inv5" not in r.text
    assert "Color matches" in r.text and "Photo taken 2 days after the invoice" in r.text  # how each compares


def test_how_it_found_this_panel(client):
    r = client.get(find(client, "vin", "silver"))
    assert "How it found this" in r.text and "What it read from each photo" in r.text
    assert "Door sticker" in r.text and "read by barcode" in r.text
    assert f"Looked up VIN {HONDA} among 5 invoices → 1 exact match" in r.text
    assert "exactly one invoice has the same confirmed VIN" in r.text


def test_nothing_fits(client):
    r = client.get(find(client, "green"))
    assert "No matching invoice found" in r.text


def test_progress_page_polls_until_done(client, monkeypatch):
    store = client.app_.state.lookups
    item = store.create(client.app_.state.lookup_dir, ["a.png"])
    r = client.get(f"/find/{item.token}/status")
    assert 'hx-trigger="every 1s"' in r.text and "Waiting to start" in r.text
    item.set_step("photo_ocr", "running")
    item.status = "running"
    r = client.get(f"/find/{item.token}/status")
    assert "Now: Reading the VIN and mileage" in r.text


def test_unknown_lookup_is_404(client):
    assert client.get("/find/nope").status_code == 404
    assert client.post("/find/nope/save", data={"invoice_id": "inv1"}).status_code == 404


# --- Save -------------------------------------------------------------------------------


def test_save_to_unlinked_invoice_links_a_new_photo_set(client):
    url = find(client, "vin", "silver")
    r = client.post(f"{url}/save", data={"invoice_id": "inv1"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/vehicles/inv1"
    with client.app_.state.sessions() as db:
        inv = db.get(Invoice, "inv1")
        c = db.query(Cluster).filter_by(invoice_id="inv1").one()
        assert inv.status == "linked" and c.status == "linked" and c.decided_by == "user"
        assert len(c.photos) == 2 and all(p.thumb_path for p in c.photos)
        [e] = db.query(MatchEvent).filter_by(cluster_id=c.id).all()
        assert (e.action, e.invoice_id, e.decided_by) == ("lookup", "inv1", "user")
    assert client.get(url).status_code == 404  # the lookup is used up


def test_save_to_already_linked_invoice_adds_the_photos(client):
    url1 = find(client, "vin")
    client.post(f"{url1}/save", data={"invoice_id": "inv1"})
    url2 = find(client, "silver")
    client.post(f"{url2}/save", data={"invoice_id": "inv1"})
    with client.app_.state.sessions() as db:
        c = db.query(Cluster).filter_by(invoice_id="inv1").one()
        assert len(c.photos) == 2
        assert [e.action for e in db.query(MatchEvent).filter_by(cluster_id=c.id)] == ["lookup", "add_photos"]


# --- Plain-language reasons -----------------------------------------------------------------


def test_explain_reasons_and_warnings():
    from app.match.cluster import Cluster as MC
    from app.match.score import InvoiceObs, score_pair

    c = MC("x", [PhotoObs("1", time=WHEN, time_from_exif=True, color="silver", shot_type="exterior"),
                 PhotoObs("2", time=WHEN, time_from_exif=True, odometer=60600, odometer_source="ocr:p",
                          shot_type="odometer")])
    inv = InvoiceObs("i", color="silver", mileage=60513, date=datetime(2026, 5, 8).date())
    reasons, warnings = lookup.explain(score_pair(c, inv), c, inv)
    assert reasons == ["color matches", "odometer within 87 mi of the invoice", "photo taken 2 days before the invoice"]
    assert warnings == []
    inv = InvoiceObs("i", color="red", mileage=90000, date=WHEN.date())
    reasons, warnings = lookup.explain(score_pair(c, inv), c, inv)
    assert reasons == ["photo taken the same day as the invoice"]
    assert warnings == ["color differs (photo silver, invoice red)", "odometer differs (60,600 vs 90,000 mi)"]


def test_store_expires_old_lookups(tmp_path):
    store = lookup.LookupStore()
    old = store.create(tmp_path, ["a.png"])
    old.created = datetime(2020, 1, 1)
    assert old.dir.exists()
    store.create(tmp_path, ["b.png"])  # creating cleans up
    assert store.get(old.token) is None and not old.dir.exists()


class Abort(BaseException):
    """Not an Exception: must still end the lookup instead of leaving it spinning."""


@pytest.mark.parametrize("error", [RuntimeError("plate reader crashed"), Abort()])
def test_a_failing_lookup_always_ends(tmp_path, error):
    def read_photos(paths, progress, uploaded_at):
        progress("alpr")
        raise error

    item = lookup.LookupStore().create(tmp_path, ["a.png"])
    item.files[0].write_bytes(png())
    if isinstance(error, Exception):
        lookup.run_lookup(item, read_photos, sessions=None, vision_on=False)
    else:
        with pytest.raises(Abort):
            lookup.run_lookup(item, read_photos, sessions=None, vision_on=False)
    assert item.status == "failed" and item.current is None
    assert item.error == ("plate reader crashed" if isinstance(error, Exception) else "Abort")
