"""Web app: pages render, review actions write match events, uploads run in the background."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.main import create_app, vin_diff
from app.models import Cluster, Invoice, Job, MatchEvent, Photo, Vehicle
from app.services import compare_rows
from app.settings import Settings


def cand(iid, score, signals=(), conflicts=(), misread=None):
    return {"invoice_id": iid, "score": score, "hard": None, "veto": None, "misread": misread,
            "signals": [{"name": n, "points": p, "detail": ""} for n, p in signals], "conflicts": list(conflicts)}


def invoice(iid, status="unlinked", vin=None, **fields):
    fields = {"vin": vin, "vendor": "Northwind Auto Body", "invoice_date": "2026-05-04", **fields}
    return Invoice(id=iid, vendor="Northwind Auto Body", invoice_no=f"N-{iid}", date=date(2026, 5, 4),
                   file_path=f"demo/invoices/{iid}.pdf", image_path=f"demo/invoices/{iid}_view.jpg", status=status,
                   extracted_json={"fields": {k: {"value": v, "confidence": 0.9, "source": "pdf_text"}
                                              for k, v in fields.items() if v is not None},
                                   "boxes": {"vin": [0.1, 0.2, 0.5, 0.25]}, "vin_trusted": True,
                                   "vin_confirmations": [{"source": "text_layer", "agrees": True, "detail": ""}],
                                   "review_reasons": []},
                   obs_json={"id": iid, "vin": vin, "vin_trusted": bool(vin), "make": "Honda", "model": "Accord",
                             "year": 2003, "color": "silver"})


SUMMARY = {"vin": "1HGCM82633A004352", "vin_trusted": True, "plate": "7KJD221", "plate_state": "CA",
           "plate_trusted": True, "color": "silver", "odometer": 60513, "time": "2026-05-04T09:00:00",
           "time_from_exif": True, "uploaders": ["ana"], "make": "HONDA", "model": "Accord", "year": 2003}


@pytest.fixture
def client(tmp_path):
    settings = Settings(tmp_path, f"sqlite:///{tmp_path / 'test.db'}")
    calls = []

    def fake_process(ctx, job_id):
        calls.append(job_id)
        with ctx.sessions() as db:
            job = db.get(Job, job_id)
            job.steps_json = [dict(s, status="done") for s in job.steps_json]
            job.status = "done"
            db.commit()

    app = create_app(settings, process=fake_process, vision_factory=lambda: None)
    with app.state.sessions() as db:
        db.add_all([
            invoice("inv1", vin="1HGCM82633A0O4352", color="silver"),  # one character misread
            invoice("inv2", color="red"),
            invoice("inv3", status="linked", vin="5YJ3E1EA4JF451621"),
            invoice("inv4"),
        ])
        db.add_all([
            Cluster(id="c1", status="review", score=18, summary_json=SUMMARY,
                    candidates_json=[cand("inv1", 18, [("color", 8), ("date", 10)], ["VIN x vs invoice y"],
                                          misread="invoice VIN likely misread: ..."),
                                     cand("inv2", 10, [("date", 10)], ["color silver vs invoice red"])],
                    reasons_json=["invoice VIN likely misread: 1HGCM82633A0O4352 vs photo 1HGCM82633A004352",
                                  "veto inv9: VIN a vs b"], truth_invoice_id="inv1"),
            Cluster(id="c2", status="review", summary_json={}, candidates_json=[],
                    reasons_json=["This photo fits two clusters"], truth_invoice_id=""),
            Cluster(id="c3", status="linked", invoice_id="inv3", decided_by="auto", how="vin", score=130,
                    summary_json=SUMMARY, truth_invoice_id="inv3"),
            Cluster(id="c4", status="unmatched", summary_json=SUMMARY, reasons_json=["best score 18 below 40"],
                    truth_invoice_id=""),
        ])
        db.add_all([Photo(cluster_id=cid, file_path=f"demo/photos/{cid}.jpg", thumb_path=f"demo/thumbs/{cid}.jpg",
                          shot_type="vin_plate", read_vin=SUMMARY["vin"],
                          extracted_json={"fields": {"vin": {"value": SUMMARY["vin"], "source": "barcode"}}})
                    for cid in ("c1", "c2", "c3", "c4")])
        db.commit()
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.calls = calls
    c.sessions = app.state.sessions
    c.app_ = app
    return c


def events(client, cid):
    with client.sessions() as db:
        return db.query(MatchEvent).filter_by(cluster_id=cid).all()


def get(client, model, key):
    with client.sessions() as db:
        return db.get(model, key)


# --- Pages ---------------------------------------------------------------------------


@pytest.mark.parametrize("url, text", [
    ("/", "Match car photos to their invoices automatically"),
    ("/vehicles", "Matched automatically"),
    ("/vehicles?show=no_photos", "N-inv4"),
    ("/vehicles?show=no_invoice", "No invoice found yet"),
    ("/vehicles/inv3", "What the invoice says"),
    ("/vehicles/inv1", "No photos yet"),
    ("/sets/c4", "no invoice fits them yet"),
    ("/review/c1", "Is this the same car?"),
    ("/review/c2", "Which invoice belongs to this car?"),
    ("/results", "How well it works"),
    ("/upload", "Add one invoice and the photos of its car"),
])
def test_pages_render(client, url, text):
    r = client.get(url)
    assert r.status_code == 200, r.text[:500]
    assert text in r.text


def test_unknown_items_are_404(client):
    assert client.get("/vehicles/nope").status_code == 404
    assert client.get("/review/nope").status_code == 404
    assert client.get("/upload/999").status_code == 404


def test_overview_counts(client):
    r = client.get("/")
    assert "Wrong matches" in r.text and "Checked against the known answers" in r.text
    with client.sessions() as db:
        db.get(Cluster, "c3").truth_invoice_id = "inv9"  # pretend the auto-link was wrong
        db.commit()
    r = client.get("/")
    assert '<div class="mt-3 text-5xl font-bold tabular-nums text-emerald-800 dark:text-emerald-200">1</div>' in r.text


def test_vehicle_page_shows_sources_and_read_boxes(client):
    r = client.get("/vehicles/inv3")
    assert 'class="field-box src-pdf" data-field="vin"' in r.text and "Show where each field was read" in r.text
    assert "Read from PDF" in r.text and "VIN confirmed" in r.text
    assert "Matched automatically:</b> the VIN on the invoice and the VIN in the photos are identical" in r.text


def test_review_starts_with_items_that_have_candidates(client):
    r = client.get("/review", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/review/c1"


def test_review_highlights_agreement_conflict_and_misread(client):
    r = client.get("/review/c1")
    assert "Same" in r.text and "Different" in r.text and "Probably misread on the invoice" in r.text
    assert '<mark class="diff">O</mark>' in r.text  # the misread character
    assert "Yes, same car" in r.text and "Pick a different invoice" in r.text and "No invoice for this car" in r.text
    assert 'data-hotkey="c 1 y"' in r.text and 'data-hotkey="n"' in r.text and 'data-hotkey="2"' in r.text
    assert "1 other invoice was ruled out because the VIN or plate is different" in r.text


def test_decided_cluster_redirects_away_from_review(client):
    r = client.get("/review/c3", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/sets/c3"


# --- Review actions -------------------------------------------------------------------


def test_confirm_links_creates_vehicle_and_match_event(client):
    r = client.post("/review/c1/confirm", data={"invoice_id": "inv1"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/review/c2?done=c1"
    c, inv = get(client, Cluster, "c1"), get(client, Invoice, "inv1")
    assert (c.status, c.invoice_id, c.decided_by) == ("linked", "inv1", "user")
    assert inv.status == "linked" and inv.vehicle_id == c.vehicle_id
    with client.sessions() as db:
        assert db.get(Cluster, "c1").vehicle.vin == SUMMARY["vin"]  # trusted photo VIN beats the misread one
    [e] = events(client, "c1")
    assert (e.action, e.invoice_id, e.decided_by, e.score) == ("confirm", "inv1", "user", 18)
    assert [s["name"] for s in e.signals_json] == ["color", "date"]


def test_pick_other_candidate_or_any_unlinked_invoice(client):
    assert client.post("/review/c1/pick", data={"invoice_id": "inv2"}).status_code == 200
    assert get(client, Cluster, "c1").invoice_id == "inv2"
    assert client.post("/review/c2/pick", data={"invoice_id": "inv4"}, follow_redirects=False).status_code == 303
    [e] = events(client, "c2")
    assert (e.action, e.invoice_id, e.score) == ("pick", "inv4", None)


def test_no_invoice(client):
    r = client.post("/review/c2/no-invoice", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/review/c1?done=c2"
    assert get(client, Cluster, "c2").status == "no_invoice"
    [e] = events(client, "c2")
    assert (e.action, e.invoice_id, e.decided_by) == ("no_invoice", None, "user")


def test_last_decision_ends_on_the_empty_queue(client):
    client.post("/review/c1/no-invoice")
    r = client.post("/review/c2/no-invoice", follow_redirects=True)
    assert r.url.path == "/review" and "Nothing needs your check right now" in r.text


def test_linking_an_already_linked_invoice_is_refused(client):
    r = client.post("/review/c1/pick", data={"invoice_id": "inv3"})
    assert r.status_code == 409 and "already linked" in r.text
    assert get(client, Cluster, "c1").status == "review" and events(client, "c1") == []


def test_actions_on_a_decided_cluster_are_refused(client):
    r = client.post("/review/c3/no-invoice")
    assert r.status_code == 409
    assert get(client, Cluster, "c3").status == "linked" and events(client, "c3") == []
    assert client.post("/review/c1/confirm", data={"invoice_id": "nope"}).status_code == 409


def test_htmx_actions_answer_with_hx_location(client):
    r = client.post("/review/c2/no-invoice", headers={"HX-Request": "true"})
    assert r.status_code == 204 and r.headers["HX-Location"] == "/review/c1?done=c2"


def test_review_count_in_nav_follows_decisions(client):
    assert '">2</span>' in client.get("/results").text
    client.post("/review/c2/no-invoice")
    assert '">1</span>' in client.get("/results").text


# --- Upload ---------------------------------------------------------------------------


def wait_for_jobs(client):
    client.app_.state.runner.executor.submit(lambda: None).result(timeout=10)  # one worker: runs after the job


def test_upload_saves_files_and_runs_in_the_background(client, tmp_path):
    files = [("invoice", ("inv.pdf", b"%PDF-1.4 test", "application/pdf")),
             ("photos", ("a b.jpg", b"jpg1", "image/jpeg")), ("photos", ("c.png", b"png", "image/png"))]
    r = client.post("/upload", files=files, data={"uploader": "Sam"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/upload/1"
    wait_for_jobs(client)
    assert client.calls == [1]
    job = get(client, Job, 1)
    assert job.status == "done" and job.uploader == "Sam"
    assert job.invoice_path == "uploads/1/invoice.pdf"
    assert job.photo_paths == ["uploads/1/photos/01_a_b.jpg", "uploads/1/photos/02_c.png"]
    assert (tmp_path / "media" / "uploads" / "1" / "photos" / "01_a_b.jpg").read_bytes() == b"jpg1"
    page = client.get("/upload/1/status")
    assert "All steps finished" in page.text and "hx-trigger" not in page.text  # stops polling


def test_upload_status_polls_while_running(client):
    with client.sessions() as db:
        db.add(Job(steps_json=jobs.new_steps(), invoice_path="x.pdf", photo_paths=[], status="running"))
        db.commit()
    r = client.get("/upload/1/status")
    assert 'hx-trigger="every 1s"' in r.text and "Step 1 of 7" in r.text and "Waiting to start" in r.text


@pytest.mark.parametrize("files, error", [
    ([("invoice", ("inv.exe", b"x", "application/octet-stream"))], "must be a PDF or an image"),
    ([("invoice", ("inv.pdf", b"x", "application/pdf")), ("photos", ("p.txt", b"x", "text/plain"))],
     "Photos must be images"),
    ([("invoice", ("inv.pdf", b"", "application/pdf"))], "Empty file"),
])
def test_upload_rejects_bad_files(client, files, error):
    r = client.post("/upload", files=files)
    assert r.status_code == 400 and error in r.text and client.calls == []


def test_failed_job_shows_the_error(tmp_path):
    def boom(ctx, job_id):
        raise RuntimeError("OCR worker died")

    app = create_app(Settings(tmp_path, f"sqlite:///{tmp_path / 't.db'}"), process=boom, vision_factory=lambda: None)
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.post("/upload", files=[("invoice", ("i.pdf", b"%PDF", "application/pdf"))])
    app.state.runner.executor.submit(lambda: None).result(timeout=10)
    r = c.get("/upload/1/status")
    assert "Something went wrong" in r.text and "RuntimeError: OCR worker died" in r.text


def test_media_serves_only_the_media_folder(client):
    assert client.get("/media/../test.db").status_code == 404
    assert client.get("/media/%2e%2e/test.db").status_code == 404


# --- Helpers ----------------------------------------------------------------------------


def test_vin_diff_marks_only_near_misses():
    assert str(vin_diff("1HGCM82633A004352", "1HGCM82633A0O4352")).count("<mark") == 1
    assert "<mark" not in str(vin_diff("1HGCM82633A004352", "5YJ3E1EA4JF451621"))
    assert str(vin_diff("<b>", None)) == "&lt;b&gt;"


def test_compare_rows_states():
    c = Cluster(id="c", status="review", summary_json=SUMMARY)
    inv = invoice("i", vin="1HGCM82633A0O4352", color="red", mileage="60600")
    rows = {r["key"]: r for r in compare_rows(c, inv, cand("i", 25, [("odometer", 15), ("date", 10)],
                                                          ["VIN a vs b", "color silver vs invoice red"],
                                                          misread="invoice VIN likely misread: x"))}
    assert rows["vin"]["state"] == "conflict" and rows["vin"]["note"] == "probably misread on the invoice"
    assert rows["color"]["state"] == "conflict" and rows["odometer"]["state"] == "agree"
    assert rows["odometer"]["invoice"] == "60,600 mi" and rows["date"]["points"] == 10
    assert rows["plate"]["state"] == "none" and rows["plate"]["word"] == "Not on invoice"
    assert [rows[k]["word"] for k in ("vin", "color", "odometer", "date")] == ["Different", "Different", "Same", "Close"]
    assert rows["make_model"]["label"] == "Make & model" and rows["odometer"]["label"] == "Mileage"


# --- Undo ----------------------------------------------------------------------------------


def test_next_page_offers_undo_right_after_a_decision(client):
    r = client.post("/review/c1/confirm", data={"invoice_id": "inv1"}, follow_redirects=False)
    assert r.headers["location"] == "/review/c2?done=c1"
    page = client.get(r.headers["location"]).text
    assert "Linked photo set c1 to invoice inv1" in page
    assert 'action="/review/c1/undo"' in page and 'data-hotkey="u"' in page
    assert 'action="/review/c1/undo"' not in client.get("/review/c2").text  # only right after


def test_undo_confirm_restores_the_review_item_and_keeps_history(client):
    client.post("/review/c1/confirm", data={"invoice_id": "inv1"})
    r = client.post("/review/c1/undo", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/review/c1"
    c, inv = get(client, Cluster, "c1"), get(client, Invoice, "inv1")
    assert (c.status, c.invoice_id, c.vehicle_id, c.decided_by) == ("review", None, None, None)
    assert (inv.status, inv.vehicle_id) == ("unlinked", None)
    assert [(e.action, e.invoice_id, e.decided_by) for e in events(client, "c1")] == [
        ("confirm", "inv1", "user"), ("undo", "inv1", "user")]
    with client.sessions() as db:
        assert db.query(Vehicle).count() == 0  # the vehicle record made by the link goes; events stay
    client.post("/review/c1/pick", data={"invoice_id": "inv2"})  # can be decided again
    assert get(client, Cluster, "c1").invoice_id == "inv2"


def test_undo_no_invoice_and_pick(client):
    client.post("/review/c2/no-invoice")
    client.post("/review/c2/undo")
    assert get(client, Cluster, "c2").status == "review"
    client.post("/review/c1/pick", data={"invoice_id": "inv2"})
    client.post("/review/c1/undo")
    assert get(client, Invoice, "inv2").status == "unlinked"
    assert [e.action for e in events(client, "c2")] == ["no_invoice", "undo"]


def test_undo_only_a_reviewer_decision_once(client):
    assert client.post("/review/c1/undo").status_code == 409  # nothing decided
    assert client.post("/review/c3/undo").status_code == 409  # an auto-link is not undone here
    client.post("/review/c2/no-invoice")
    client.post("/review/c2/undo")
    assert client.post("/review/c2/undo").status_code == 409
    assert [e.action for e in events(client, "c2")] == ["no_invoice", "undo"]


# --- Local network only -----------------------------------------------------------------


@pytest.mark.parametrize("host, ok", [("127.0.0.1", True), ("192.168.1.23", True), ("10.0.0.7", True),
                                      ("172.20.0.2", True), ("::1", True), ("fe80::1c2d", True),
                                      ("8.8.8.8", False), ("100.64.1.1", False), ("2001:4860::8888", False),
                                      ("testclient", False)])
def test_only_this_computer_and_the_local_network(tmp_path, host, ok):
    app = create_app(Settings(tmp_path, f"sqlite:///{tmp_path / 't.db'}"), vision_factory=lambda: None)
    r = TestClient(app, client=(host, 50000)).get("/")
    assert r.status_code == (200 if ok else 403)
    if not ok:
        assert "local network" in r.text
        assert TestClient(app, client=(host, 50000)).get("/static/app.js").status_code == 403


def test_reset_database_in_place_keeps_open_engines_working(tmp_path):
    from app.models import make_sessionmaker, reset_database

    url = f"sqlite:///{tmp_path / 'r.db'}"
    server = make_sessionmaker(url)  # a running app
    with server() as db:
        db.add(invoice("old"))
        db.commit()
    reset_database(url)  # tools/reset_demo.py while the app runs
    with server() as db:
        assert db.query(Invoice).count() == 0
        db.add(invoice("new"))
        db.commit()


# --- Plain language ------------------------------------------------------------------------

JARGON = ("cluster", "auto-link", "Auto-link", "Review queue", "unmatched", "Unmatched", "trusted", "Trusted",
          "candidate", "score below", "veto")


@pytest.mark.parametrize("url", ["/", "/vehicles", "/vehicles?show=no_photos", "/vehicles?show=no_invoice",
                                 "/vehicles/inv3", "/sets/c4", "/review/c1", "/review/c2", "/upload", "/find",
                                 "/results"])
def test_pages_speak_plain_language(client, url):
    import re

    html = client.get(url).text
    visible = re.sub(r"<(script|style)[\s\S]*?</\1>|<details[\s\S]*?</details>|<[^>]+>", " ", html)
    for word in JARGON:
        assert word not in visible, f"{word!r} on {url}"
    assert re.search(r"<h1[^>]*>", html)  # every page says what it is


def test_navigation_labels(client):
    html = client.get("/").text
    for label in ("Home", "Vehicles", "Needs your check", "Find invoice by photo", "Upload", "How well it works"):
        assert f">\n            {label}" in html or f">{label}<" in html


def test_plain_reasons():
    from app.services import plain_reasons

    assert plain_reasons(["invoice VIN likely misread: 1PMCU vs photo 1FMCU (3 differences)", "veto a: x", "veto b: y"]) == [
        "The VIN on the invoice is 3 characters different from the VIN in the photos. "
        "It was probably misread from the invoice.",
        "2 other invoices were ruled out because the VIN or plate is different."]
    assert plain_reasons(["score 43 below 80"]) == ["Some details match, but not enough to be sure it's the same car."]
    assert plain_reasons(["This photo fits two clusters (same uploader, minutes apart)"]) == [
        "This photo could belong to either of two cars photographed minutes apart."]
    assert plain_reasons(["invoice plate likely misread: EN90 vs photo EN90604 (cut short)"]) == [
        "The license plate on the invoice (EN90) looks cut short. The photos show EN90604."]


def test_long_invoice_lists_show_the_newest_first_page(client, monkeypatch):
    import app.main as m

    with client.sessions() as db:
        db.add_all([invoice(f"bulk{n:03d}") for n in range(70)])
        db.commit()
    r = client.get("/vehicles?show=no_photos")
    assert r.text.count('href="/vehicles/bulk') + r.text.count('href="/vehicles/inv') == m.PAGE
    assert f"Showing the newest {m.PAGE} of 73 invoices" in r.text and "limit=120" in r.text
    assert '<span class="tabular-nums opacity-70">73</span>' in r.text  # the tab counts all of them
    r = client.get("/vehicles?show=no_photos&limit=120")
    assert "Show more" not in r.text
