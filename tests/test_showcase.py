"""Showcase mode (DEMO_MODE=showcase): the read-only public demo served from a prebuilt bundle."""

import json
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import showcase
from app.main import create_app
from app.models import Cluster, Invoice, Photo, make_sessionmaker
from app.settings import ROOT, Settings
from tests.test_find import HONDA, SEEN, fields, obs_invoice, png
from tests.test_web import SUMMARY, cand

sys.path.insert(0, str(ROOT / "tools"))

import build_showcase as bs  # noqa: E402

BANNER = "Showcase mode: results were produced by the same pipeline running locally."
CREDIT = {"photo": "honda_cr-v_silver_1.jpg", "title": "File:2018 Honda CR-V.jpg", "author": "Jane Doe",
          "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
          "page_url": "https://commons.wikimedia.org/wiki/File:2018_Honda_CR-V.jpg", "used_for": ["car1-outside"]}


def fake_reader(paths, progress, uploaded_at):
    for step in ("vision", "ocr", "alpr"):
        progress(step)
    return [fields(p, **SEEN[p.stem.rsplit("_", 1)[1]]) for p in paths], None


def source_db(path: Path) -> Path:
    """A small processed demo database: invoices, a review item, a linked set."""
    sessions = make_sessionmaker(f"sqlite:///{path}")
    with sessions() as db:
        db.add_all([
            obs_invoice("inv1", vin=HONDA, vin_trusted=True, color="silver", make="Honda", model="Accord"),
            obs_invoice("inv2", color="silver", make="Toyota", model="Camry"),
            obs_invoice("inv3", color="silver"),
            obs_invoice("inv5", color="red"),
        ])
        db.add_all([
            Cluster(id="c1", status="review", score=18, summary_json=SUMMARY, truth_invoice_id="inv2",
                    candidates_json=[cand("inv2", 18, [("color", 8), ("date", 10)]), cand("inv3", 10, [("date", 10)])],
                    reasons_json=["best score 18 below 40"]),
            Cluster(id="c2", status="review", summary_json={}, candidates_json=[], reasons_json=["fits two"]),
        ])
        db.add_all([Photo(cluster_id=cid, file_path=f"demo/photos/{cid}.jpg", thumb_path=f"demo/thumbs/{cid}.jpg",
                          shot_type="exterior", extracted_json={"fields": {}}) for cid in ("c1", "c2")])
        db.commit()
    return path


@pytest.fixture
def bundle(tmp_path) -> Path:
    """A bundle as tools/build_showcase.py writes it, from real (fake-reader) lookups."""
    db_file = source_db(tmp_path / "src.db")
    media = tmp_path / "media"
    for key in ("demo/photos/c1.jpg", "demo/thumbs/c1.jpg", "demo/photos/c2.jpg", "demo/thumbs/c2.jpg",
                "demo/invoices/inv1_view.jpg", "demo/invoices/inv1.pdf", "demo/sample/invoice.jpg",
                "uploads/1/invoice.jpg"):
        (media / key).parent.mkdir(parents=True, exist_ok=True)
        (media / key).write_bytes(png() if key.endswith(".jpg") else b"%PDF-1.4")
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("car1_vin.png", "car1_silver.png"):
        (photos / name).write_bytes(png())
    sessions = make_sessionmaker(f"sqlite:///{db_file}")
    work = tmp_path / "work"
    samples = [
        bs.run_sample(sessions, fake_reader, "car1-vin", "car1", "VIN sticker", [photos / "car1_vin.png"], True, work),
        bs.run_sample(sessions, fake_reader, "car1-outside", "car1", "Outside", [photos / "car1_silver.png"], True,
                      work),
        bs.run_sample(sessions, fake_reader, "car1-all", "car1", "All photos",
                      [photos / "car1_vin.png", photos / "car1_silver.png"], True, work),
    ]
    sessions.kw["bind"].dispose()
    out = tmp_path / "showcase"
    cars = [{"id": "car1", "title": "Silver Honda Accord", "note": "Its invoice has no photos yet."}]
    bs.write_bundle(out, db_file, media, samples, cars, [CREDIT], {"vision_model": "qwen3.5:4b-q4_K_M"})
    return out


@pytest.fixture
def make_client(tmp_path, bundle, monkeypatch):
    monkeypatch.delenv("SHOWCASE_RATE_LIMIT", raising=False)
    monkeypatch.delenv("SHOWCASE_POST_LIMIT", raising=False)

    def make(host="8.8.8.8"):
        settings = Settings(tmp_path / "state", "sqlite:///unused.db", demo_mode="showcase", showcase_dir=bundle)
        app = create_app(settings, vision_factory=lambda: None)
        c = TestClient(app, client=(host, 50000))
        c.app_ = app
        return c

    return make


@pytest.fixture
def client(make_client):
    return make_client()


def base_rows(bundle: Path, sql: str) -> list:
    con = sqlite3.connect(f"file:{bundle / 'app.db'}?mode=ro", uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# --- Pages, banner, public visitors -------------------------------------------------------


@pytest.mark.parametrize("url", ["/", "/vehicles", "/review/c1", "/find", "/find/car1-vin", "/results", "/upload",
                                 "/credits"])
def test_pages_render_with_the_banner_for_public_visitors(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text[:300]
    assert BANNER in r.text
    assert "Run it yourself with Docker" in r.text


def test_full_mode_has_no_banner_and_stays_local_only(tmp_path):
    app = create_app(Settings(tmp_path, f"sqlite:///{tmp_path / 't.db'}"), vision_factory=lambda: None)
    assert BANNER not in TestClient(app, client=("127.0.0.1", 50000)).get("/").text
    assert TestClient(app, client=("8.8.8.8", 50000)).get("/").status_code == 403


def test_health_check(client, tmp_path):
    assert client.get("/healthz").json() == {"status": "ok", "mode": "showcase"}
    app = create_app(Settings(tmp_path, f"sqlite:///{tmp_path / 't.db'}"), vision_factory=lambda: None)
    assert TestClient(app, client=("127.0.0.1", 50000)).get("/healthz").json() == {"status": "ok", "mode": "full"}


def test_unknown_demo_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="DEMO_MODE"):
        create_app(Settings(tmp_path, f"sqlite:///{tmp_path / 't.db'}", demo_mode="party"))


def test_missing_bundle_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_showcase.py"):
        create_app(Settings(tmp_path, "sqlite:///x.db", demo_mode="showcase", showcase_dir=tmp_path / "nope"))


# --- Find: gallery and saved results -----------------------------------------------------


def test_find_page_is_a_gallery_of_sample_photos(client):
    r = client.get("/find")
    assert r.text.count("Try this photo") == 2  # the single-photo samples
    assert "Try all 2 photos" in r.text
    assert 'href="/find/car1-vin"' in r.text and 'href="/find/car1-all"' in r.text
    assert "Silver Honda Accord" in r.text and "Its invoice has no photos yet." in r.text
    assert 'type="file"' not in r.text  # no uploads
    assert client.get("/find/car1-vin/thumb/0.jpg").headers["content-type"] == "image/jpeg"


def test_saved_result_is_shown_with_how_it_found_this(client):
    r = client.get("/find/car1-vin")
    assert r.status_code == 200
    assert "Found it: VIN matches" in r.text
    assert "How it found this" in r.text and "What it read from each photo" in r.text
    assert f"Looked up VIN {HONDA} among 4 invoices" in r.text
    assert "hx-trigger" not in r.text  # finished: nothing to poll
    assert "qwen3.5:4b-q4_K_M" in r.text  # how it was produced
    many = client.get("/find/car1-outside").text
    assert "invoices fit this description" in many


def test_save_to_invoice_is_disabled_with_a_note(client, bundle):
    r = client.get("/find/car1-vin")
    assert 'action="/find/car1-vin/save"' not in r.text
    assert "Save to this invoice" in r.text and "disabled" in r.text
    assert "Saving is turned off in the showcase" in r.text
    r = client.post("/find/car1-vin/save", data={"invoice_id": "inv1"})
    assert r.status_code == 403 and "Docker" in r.text
    assert base_rows(bundle, "select count(*) from photos") == [(2,)]


def test_unknown_sample_is_404(client):
    assert client.get("/find/nope").status_code == 404


@pytest.mark.parametrize("url", ["/find", "/upload", "/upload/sample"])
def test_uploads_are_refused_before_reading_the_body(client, url):
    r = client.post(url, files=[("photos", ("a.png", png() * 50, "image/png"))])
    assert r.status_code == 403
    assert "Docker" in r.text


def test_upload_page_points_to_docker(client):
    r = client.get("/upload")
    assert 'type="file"' not in r.text and "Process the sample car" not in r.text
    assert "Uploading is turned off in the showcase" in r.text


# --- Review: decisions last for the visitor's session only ---------------------------------


def test_review_decisions_stay_in_the_visitors_session(make_client, bundle):
    alice, bob = make_client(), make_client("1.1.1.1")
    r = alice.post("/review/c1/confirm", data={"invoice_id": "inv2"}, follow_redirects=False)
    assert r.status_code == 303
    assert "vm_visitor" in r.cookies or alice.cookies.get("vm_visitor")
    assert alice.get("/sets/c1", follow_redirects=False).headers["location"] == "/vehicles/inv2"
    assert "Linked photo set c1 to invoice inv2" in alice.get("/review/c2?done=c1").text
    # Another visitor, and the shared database, never see it.
    assert bob.get("/review/c1").status_code == 200
    assert base_rows(bundle, "select status from clusters where id='c1'") == [("review",)]
    assert base_rows(bundle, "select count(*) from match_events") == [(0,)]
    # Undo works within the session.
    alice.post("/review/c1/undo")
    assert alice.get("/review/c1").status_code == 200


def test_visitor_without_a_session_reads_the_shared_database(client):
    client.get("/review/c1")
    assert not client.cookies.get("vm_visitor")  # looking creates nothing


def test_shared_database_is_read_only(bundle, tmp_path):
    visitors = showcase.VisitorDatabases(bundle / "app.db", tmp_path / "v")
    with visitors.base() as db:
        db.get(Invoice, "inv1").vendor = "changed"
        with pytest.raises(Exception, match="readonly"):
            db.commit()


def test_oldest_visitor_sessions_are_dropped(bundle, tmp_path):
    visitors = showcase.VisitorDatabases(bundle / "app.db", tmp_path / "v", max_visitors=2)
    a, b = visitors.create(), visitors.create()
    assert visitors.get(a) is not None
    c = visitors.create()  # b is now the least recently used
    assert visitors.get(b) is None and visitors.get(a) is not None and visitors.get(c) is not None
    assert len(list((tmp_path / "v").glob("*.db"))) == 2


def test_idle_visitor_sessions_expire(bundle, tmp_path):
    visitors = showcase.VisitorDatabases(bundle / "app.db", tmp_path / "v", idle=timedelta(hours=2))
    a = visitors.create()
    visitors.cleanup(now=datetime.now() + timedelta(hours=3))
    assert visitors.get(a) is None
    assert not list((tmp_path / "v").glob("*.db"))


def test_forged_visitor_cookie_is_ignored(client):
    client.cookies.set("vm_visitor", "../../etc/passwd")
    assert client.get("/review/c1").status_code == 200


# --- Public internet safety ----------------------------------------------------------------


def test_rate_limiter_window():
    rl = showcase.RateLimiter(limit=3, window=60)
    t = 1000.0
    assert [rl.allow("a", t), rl.allow("a", t + 1), rl.allow("a", t + 2), rl.allow("a", t + 3)] == [
        True, True, True, False]
    assert rl.allow("b", t + 3)  # per visitor
    assert rl.allow("a", t + 61)  # the window moved on


def test_rate_limit_answers_429(make_client, monkeypatch):
    monkeypatch.setenv("SHOWCASE_RATE_LIMIT", "3")
    c = make_client()
    assert [c.get("/").status_code for _ in range(4)] == [200, 200, 200, 429]
    assert c.get("/static/app.js").status_code == 200  # static files don't count
    assert make_client("1.1.1.1").get("/").status_code == 200


def test_post_limit_is_stricter(make_client, monkeypatch):
    monkeypatch.setenv("SHOWCASE_POST_LIMIT", "1")
    c = make_client()
    assert c.post("/review/c2/no-invoice", follow_redirects=False).status_code == 303
    assert c.post("/review/c2/undo", follow_redirects=False).status_code == 429


def test_large_requests_are_refused(client):
    r = client.post("/review/c1/confirm", data={"invoice_id": "x" * 100_000})
    assert r.status_code == 413


@pytest.mark.parametrize("url", ["/media/demo/sample/invoice.jpg", "/media/uploads/1/invoice.jpg",
                                 "/media/../app.db", "/media/%2e%2e/app.db", "/app.db", "/samples.json"])
def test_no_answer_keys_private_or_bundle_files_are_served(client, url):
    assert client.get(url).status_code == 404


def test_only_the_referenced_media_is_in_the_bundle(bundle):
    files = {p.relative_to(bundle / "media").as_posix() for p in (bundle / "media").rglob("*") if p.is_file()}
    assert files == {"demo/photos/c1.jpg", "demo/thumbs/c1.jpg", "demo/photos/c2.jpg", "demo/thumbs/c2.jpg",
                     "demo/invoices/inv1_view.jpg", "demo/invoices/inv1.pdf"}


def test_credits_page_lists_the_wikimedia_photos(client):
    r = client.get("/credits")
    for text in ("2018 Honda CR-V", "Jane Doe", "CC BY-SA 4.0", CREDIT["page_url"], CREDIT["license_url"]):
        assert text in r.text
    assert 'href="/credits"' in client.get("/").text


def test_showcase_never_loads_ocr_plate_reader_or_vision(bundle, tmp_path):
    """Pages and saved lookups work without the heavy engines (and without Pillow)."""
    code = textwrap.dedent(f"""
        import sys
        for name in ("paddleocr", "paddle", "fast_alpr", "cv2", "onnxruntime", "PIL", "pdfplumber", "pypdfium2"):
            sys.modules[name] = None  # importing any of them now fails
        from fastapi.testclient import TestClient
        from app.main import create_app
        from app.settings import Settings
        s = Settings({str(tmp_path / 'st')!r}, "sqlite:///unused.db", demo_mode="showcase",
                     showcase_dir={str(bundle)!r})
        c = TestClient(create_app(s), client=("8.8.8.8", 1))
        for url in ("/", "/vehicles", "/review/c1", "/find", "/find/car1-vin", "/find/car1-all", "/results",
                    "/upload", "/credits", "/vehicles/inv1"):
            assert c.get(url).status_code == 200, url
        assert c.post("/review/c1/confirm", data={{"invoice_id": "inv2"}}).status_code == 200
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip().endswith("ok"), r.stderr[-2000:]


def test_bundle_metadata_round_trip(bundle):
    b = showcase.Bundle.load(bundle)
    assert [s.id for s in b.samples.values()] == ["car1-vin", "car1-outside", "car1-all"]
    s = b.samples["car1-all"]
    assert s.names == ["car1_vin.png", "car1_silver.png"] and s.seconds >= 0
    assert (bundle / "samples" / "car1-all" / "thumb_01.jpg").exists()
    assert b.meta["vision_model"] == "qwen3.5:4b-q4_K_M"
    item = s.lookup()
    assert item.status == "done" and item.result["kind"] == "confident"
    data = json.loads((bundle / "samples.json").read_text())
    assert "truth" not in json.dumps(data) and "answer" not in json.dumps(data).lower()
