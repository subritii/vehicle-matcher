"""Vision-model gap filling. Ollama is mocked: the client tests use an httpx mock
transport, the gap-filling tests a fake model that returns fixed fields."""

import json

import httpx
import pytest
from PIL import Image

from app.extract import invoice as inv
from app.extract import vision
from app.extract.invoice import Page, extract_from_page, fill_gaps_with_vision
from tests.test_invoice import HONDA, FakeVpic, make_vin, ocr_page

FORD = make_vin("1FTEW1EP_JF654321")  # 2018 Ford, valid check digit
BAD_CHECK = HONDA[:8] + ("0" if HONDA[8] != "0" else "1") + HONDA[9:]


class FakeVision:
    name = "vision:fake"

    def __init__(self, **values):
        self.values = values
        self.calls = 0
        self.log = None

    def read_invoice(self, images):
        self.calls += 1
        if self.log is not None:
            self.log.append("vision")
        return dict(self.values)

    def unload(self):
        if self.log is not None:
            self.log.append("unload")


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "scan.png"
    Image.new("RGB", (40, 30), "white").save(p)
    return p


def fill(text, image_path, vpic=False, **values):
    f = inv._extract_fields(ocr_page(text))
    model = FakeVision(**values)
    return fill_gaps_with_vision(f, image_path, model, vpic=vpic), model


# --- Ollama client -----------------------------------------------------------------


def ollama(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test")


def test_client_sends_image_prompt_schema_and_keep_alive_zero():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps({"vin": HONDA, "plate": None})}})

    model = vision.VisionModel("tiny-vl", client=ollama(handler))
    out = model.read_invoice([Image.new("RGB", (3000, 1500), "white")])
    assert out["vin"] == HONDA and out["plate"] is None
    [body] = seen
    assert body["model"] == "tiny-vl"
    assert body["keep_alive"] == 0
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert set(body["format"]["properties"]) == set(inv.FIELD_NAMES)
    [msg] = body["messages"]
    assert "invoice" in msg["content"].lower()
    [img] = msg["images"]
    import base64, io
    with Image.open(io.BytesIO(base64.b64decode(img))) as im:
        assert max(im.size) == model.max_side  # downscaled before sending


def test_client_retries_without_think_for_models_that_reject_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": '"tiny-vl" does not support thinking'})
        return httpx.Response(200, json={"message": {"content": "{}"}})

    assert vision.VisionModel("tiny-vl", client=ollama(handler)).read_invoice([Image.new("RGB", (9, 9))]) == {}
    assert ["think" in b for b in bodies] == [True, False]


@pytest.mark.parametrize("response", [
    httpx.Response(500, json={"error": "model runner crashed"}),
    httpx.Response(200, json={"message": {"content": "not json"}}),
    httpx.Response(200, json={"message": {"content": "[1, 2]"}}),
])
def test_client_errors_raise_vision_error(response):
    model = vision.VisionModel("tiny-vl", client=ollama(lambda r: response))
    with pytest.raises(vision.VisionError):
        model.read_invoice([Image.new("RGB", (9, 9))])


def test_session_keeps_model_loaded_then_unloads():
    bodies = []

    def handler(request):
        bodies.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    model = vision.VisionModel("tiny-vl", client=ollama(handler))
    with model.session():
        model.read_invoice([Image.new("RGB", (9, 9))])
        model.read_invoice([Image.new("RGB", (9, 9))])
    assert [b["keep_alive"] for _, b in bodies[:2]] == ["10m", "10m"]
    path, last = bodies[-1]
    assert path == "/api/generate" and last == {"model": "tiny-vl", "keep_alive": 0}
    assert model.keep_alive == 0


def test_from_env_is_none_without_ollama_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "")
    assert vision.from_env() is None
    monkeypatch.setenv("OLLAMA_MODEL", "tiny-vl")
    monkeypatch.setenv("OLLAMA_HOST", "http://box:1234")
    m = vision.from_env()
    assert (m.model, m.host, m.name) == ("tiny-vl", "http://box:1234", "vision:tiny-vl")


# --- Filling gaps ------------------------------------------------------------------


def test_blank_fields_are_filled_and_marked_as_vision(image_path):
    f, _ = fill("ACME AUTO\nPlate: 7KJD221 CA\n", image_path,
                mileage="84,211", color="Silver Metallic", invoice_date="03/14/2025", total="$514.50",
                year="2003", make="HONDA", model="Accord")
    assert f.value("mileage") == 84211
    assert f.value("color") == "silver"
    assert str(f.value("invoice_date")) == "2025-03-14"
    assert str(f.value("total")) == "514.50"
    assert (f.value("year"), f.value("make"), f.value("model")) == (2003, "Honda", "Accord")
    for name in ("mileage", "color", "invoice_date", "total", "year", "make", "model"):
        assert f.fields[name].source == "vision:fake"
        assert 0 < f.fields[name].confidence < 1
    assert f.fields["plate"].source == "ocr:test"
    assert f.to_dict()["fields"]["mileage"]["source"] == "vision:fake"


def test_confident_rule_values_are_never_overwritten(image_path):
    f, _ = fill("ACME AUTO\nPlate: 7KJD221 CA\nOdometer: 84,211 mi\n", image_path,
                plate="9ZZZ999", plate_state="NV", mileage="99999")
    assert (f.value("plate"), f.value("plate_state"), f.value("mileage")) == ("7KJD221", "CA", 84211)


def test_low_confidence_rule_value_is_replaced(image_path):
    f = inv._extract_fields(ocr_page("ACME AUTO\nColor: Blk\n"))
    f.fields["color"].confidence = 0.2
    fill_gaps_with_vision(f, image_path, FakeVision(color="Blue"), vpic=False)
    assert f.value("color") == "blue" and f.fields["color"].source == "vision:fake"


# Fields with no check digit: a disagreement is sent to review, not decided.
CONFLICTS = [
    ("Odometer: 84,211 mi", "mileage", "84,217", 84211),
    ("Invoice Date: 03/14/2025", "invoice_date", "03/14/2023", "2025-03-14"),
    ("Invoice No: 2025-00417", "invoice_no", "2025-00411", "2025-00417"),
    ("Total Due $514.50", "total", "$574.50", "514.50"),
]


@pytest.mark.parametrize("line, name, model_value, kept", CONFLICTS)
def test_unchecked_field_disagreement_goes_to_review(image_path, line, name, model_value, kept):
    f = inv._extract_fields(ocr_page(f"ACME AUTO\n{line}\n"))
    f.fields[name].confidence = 0.2
    fill_gaps_with_vision(f, image_path, FakeVision(**{name: model_value}), vpic=False)
    assert str(f.value(name)) == str(kept) and f.fields[name].source == "ocr:test"
    assert f.needs_review
    [reason] = [r for r in f.review_reasons if name in r]
    assert "vision model read" in reason
    assert f.to_dict()["conflicts"][name]["vision"] == model_value


@pytest.mark.parametrize("line, name, model_value, kept", CONFLICTS)
def test_unchecked_field_agreement_is_not_a_conflict(image_path, line, name, model_value, kept):
    f = inv._extract_fields(ocr_page(f"ACME AUTO\n{line}\n"))
    f.fields[name].confidence = 0.2
    same = {"mileage": "84211", "invoice_date": "2025-03-14", "invoice_no": "2025-00417", "total": "514.50"}[name]
    fill_gaps_with_vision(f, image_path, FakeVision(**{name: same}), vpic=False)
    assert str(f.value(name)) == str(kept)
    assert not any(name in r for r in f.review_reasons)


@pytest.mark.parametrize("name, value", [
    ("invoice_date", "2099-01-01"),  # future
    ("invoice_date", "1998-06-30"),  # before 2000
    ("mileage", "1,250,000"),
    ("total", "0.00"),
    ("total", "-45.00"),
])
def test_implausible_model_values_are_dropped(image_path, name, value):
    f, _ = fill("ACME AUTO\n", image_path, **{name: value})
    assert name not in f.fields


@pytest.mark.parametrize("line, reason", [
    ("Invoice Date: 01/05/2099", "invoice_date 2099-01-05 is in the future"),
    ("Invoice Date: 01/05/1998", "invoice_date 1998-01-05 is before 2000"),
    ("Total Due $0.00", "total 0.00 is not above zero"),
])
def test_implausible_rule_values_go_to_review(line, reason):
    f = extract_from_page(ocr_page(f"ACME AUTO\nPlate: 7KJD221 CA\n{line}\n"), vpic=False)
    assert reason in f.review_reasons


def test_unparseable_or_invented_values_are_dropped(image_path):
    f, _ = fill("ACME AUTO\n2003 Honda Accord\n", image_path,
                color="sparkly", mileage="about 80k", invoice_date="last week", total="n/a")
    for name in ("color", "mileage", "invoice_date", "total"):
        assert name not in f.fields


def test_model_name_must_be_a_known_model_for_the_make(image_path):
    f, _ = fill("ACME AUTO\n", image_path, year="2003", make="Honda", model="Skyrocket")
    assert f.value("make") == "Honda"
    assert "model" not in f.fields


def test_vision_state_is_not_attached_to_an_ocr_plate(image_path):
    f, _ = fill("ACME AUTO\nPlate: 7KJD221\n", image_path, plate="7KJD221", plate_state="CA")
    assert f.value("plate") == "7KJD221" and "plate_state" not in f.fields
    assert f.plate_trusted


def without_plate(text):
    """Rules result for ``text`` with the plate the OCR read removed (as if it were unreadable)."""
    f = inv._extract_fields(ocr_page(text))
    f.fields.pop("plate", None)
    f.fields.pop("plate_state", None)
    return f


def test_model_plate_kept_near_a_plate_label_but_never_trusted(image_path):
    f = without_plate("ACME AUTO\nLic Plate: 7KJ0Z21 CA\n")
    fill_gaps_with_vision(f, image_path, FakeVision(plate="7KJD221", plate_state="ca"), vpic=False)
    assert (f.value("plate"), f.value("plate_state")) == ("7KJD221", "CA")
    assert f.fields["plate"].source == f.fields["plate_state"].source == "vision:fake"
    assert not f.plate_trusted
    assert f.to_dict()["plate_trusted"] is False


@pytest.mark.parametrize("label", ["Plate", "Lic", "License", "Tag"])
def test_model_plate_label_words(image_path, label):
    f = without_plate(f"ACME AUTO\n{label}\n7KJD22l\n")
    fill_gaps_with_vision(f, image_path, FakeVision(plate="7KJD221"), vpic=False)
    assert f.value("plate") == "7KJD221"


@pytest.mark.parametrize("label", ["LICA", "LICE", "P1ATE NO", "TAGIHAT"])
def test_model_plate_label_garbled_by_ocr(image_path, label):
    # "LIC#" read as "LICA": the rules can't use the label, the gate still sees it.
    f = without_plate(f"ACME AUTO\n{label}  MILEAGE IN\nWA KTS5749  60513\n")
    fill_gaps_with_vision(f, image_path, FakeVision(plate="KTS5749", plate_state="WA"), vpic=False)
    assert (f.value("plate"), f.value("plate_state")) == ("KTS5749", "WA")
    assert not f.plate_trusted


def test_label_must_start_a_word(image_path):
    f, _ = fill("ACME AUTO\nPOLICY 39624\n", image_path, plate="39624")
    assert "plate" not in f.fields


def test_model_plate_dropped_without_a_plate_label(image_path):
    f, _ = fill("ACME AUTO\nPO Box 39624\n", image_path, plate="39624")
    assert "plate" not in f.fields and not f.plate_trusted


def test_model_plate_dropped_when_label_has_nothing_like_it(image_path):
    f = without_plate("ACME AUTO\nPlate: 8BWD418 CA\nTotal 100.00\n")
    fill_gaps_with_vision(f, image_path, FakeVision(plate="39624"), vpic=False)
    assert "plate" not in f.fields


def test_digit_only_plate_is_allowed(image_path):
    f = without_plate("ACME AUTO\nTag: 396Z4 DE\n")
    fill_gaps_with_vision(f, image_path, FakeVision(plate="39624", plate_state="DE"), vpic=False)
    assert (f.value("plate"), f.value("plate_state")) == ("39624", "DE")


def test_model_is_not_called_when_nothing_is_missing():
    f = extract_from_page(Page.from_text(
        f"ACME AUTO\nInvoice No: 2025-00417  Invoice Date: 03/14/2025\nVehicle: 2003 Honda Accord EX  "
        f"Color: Silver\nVIN: {HONDA}  License: 7KJD221 CA\nOdometer: 84,211 mi\nTotal Due $514.50\n"))
    assert set(inv.FIELD_NAMES) <= set(f.fields) and f.vin_trusted
    model = FakeVision(vin=FORD)
    fill_gaps_with_vision(f, "unused.png", model, vpic=False)
    assert model.calls == 0


def test_vision_error_leaves_rules_result_unchanged(image_path):
    class Broken(FakeVision):
        def read_invoice(self, images):
            raise vision.VisionError("model runner crashed")

    f = inv._extract_fields(ocr_page("ACME AUTO\nPlate: 7KJD221 CA\n"))
    before = dict(f.fields)
    fill_gaps_with_vision(f, image_path, Broken(), vpic=False)
    assert f.fields == before
    assert any("vision model failed" in n for n in f.notes)


# --- VIN safety --------------------------------------------------------------------


OCR_TEXT = f"ACME AUTO\nVIN: {HONDA}\n2003 Honda Accord EX\nPlate: 7KJD221 CA\n"


def test_vision_vin_matching_ocr_vin_confirms_it(image_path):
    f, _ = fill(OCR_TEXT, image_path, vpic=False, vin=HONDA.lower())
    assert f.value("vin") == HONDA and f.fields["vin"].source == "ocr:test"
    assert f.vin_trusted
    assert ("vision", True) in [(c.source, c.agrees) for c in f.vin_confirmations]


def test_vision_vin_disagreeing_with_ocr_vin_blocks_trust(image_path):
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")  # vPIC alone would trust the OCR VIN
    f, _ = fill(OCR_TEXT, image_path, vpic=vpic, vin=FORD)
    assert f.value("vin") == HONDA
    assert not f.vin_trusted
    assert any("vision model read a different VIN" in r for r in f.review_reasons)


def test_vision_vin_failing_check_digit_is_ignored(image_path):
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f, _ = fill(OCR_TEXT, image_path, vpic=vpic, vin=BAD_CHECK)
    assert f.value("vin") == HONDA and f.vin_trusted
    assert "vision" not in [c.source for c in f.vin_confirmations]
    f, _ = fill("ACME AUTO\n2003 Honda Accord\n", image_path, vpic=vpic, vin=BAD_CHECK)
    assert "vin" not in f.fields


def test_model_only_vin_needs_vpic_and_printed_year_make_and_is_never_trusted(image_path):
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f, _ = fill("ACME AUTO\n2003 Honda Accord EX\nPlate: 7KJD221 CA\n", image_path, vpic=vpic, vin=HONDA)
    assert f.value("vin") == HONDA and f.fields["vin"].source == "vision:fake"
    assert vpic.calls == [HONDA]
    assert not f.vin_trusted
    assert any("only by the vision model" in r for r in f.review_reasons)


def test_model_only_vin_rejected_when_vpic_disagrees(image_path):
    f, _ = fill("ACME AUTO\n2018 Ford F-150\n", image_path,
                vpic=FakeVpic(year=2003, make="HONDA", model="Accord"), vin=HONDA)
    assert "vin" not in f.fields


def test_model_only_vin_rejected_without_vpic(image_path):
    f, _ = fill("ACME AUTO\n2003 Honda Accord EX\n", image_path, vpic=False, vin=HONDA)
    assert "vin" not in f.fields


def test_model_only_vin_not_confirmed_by_year_and_make_from_the_model(image_path):
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f, _ = fill("ACME AUTO\nPlate: 7KJD221 CA\n", image_path, vpic=vpic,
                vin=HONDA, year="2003", make="Honda", model="Accord")
    assert "vin" not in f.fields
    assert f.value("make") == "Honda"  # the other fields are still filled


def test_ocr_vin_not_confirmed_by_vpic_against_vision_year_and_make():
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f = inv._extract_fields(ocr_page(f"ACME AUTO\nVIN: {HONDA}\n"))
    f.fields["year"] = inv.FieldValue(2003, 0.5, "2003", "vision:fake")
    f.fields["make"] = inv.FieldValue("Honda", 0.5, "Honda", "vision:fake")
    inv._finalize(f, vpic)
    assert not f.vin_trusted
    [c] = f.vin_confirmations
    assert (c.source, c.agrees) == ("vpic", False) and "no printed year and make" in c.detail


def test_model_vin_replaces_ocr_vin_failing_check_digit_but_stays_untrusted(image_path):
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f, _ = fill(f"ACME AUTO\nVIN: {BAD_CHECK}\n2003 Honda Accord EX\n", image_path, vpic=vpic, vin=HONDA)
    assert f.value("vin") == HONDA and f.fields["vin"].source == "vision:fake"
    assert not f.vin_trusted


def test_text_layer_vin_is_left_alone(image_path):
    f = inv._extract_fields(Page.from_text(f"ACME AUTO\nVIN: {HONDA}\n"))
    fill_gaps_with_vision(f, image_path, FakeVision(vin=FORD), vpic=False)
    assert f.value("vin") == HONDA and f.vin_trusted


# --- OCR and the model never run together -------------------------------------------


def test_extract_invoice_stops_ocr_before_the_model_and_unloads_after(image_path):
    log = []

    class Ocr:
        name = "fake"
        word_level = False

        def __call__(self, image):
            log.append("ocr")
            return [inv.Seg("ACME AUTO", 0, 0, 100, 10, 0.95)]

        def close(self):
            log.append("close_ocr")

    model = FakeVision(mileage="1234")
    model.log = log
    f = inv.extract_invoice(image_path, ocr=Ocr(), vpic=False, vision=model)
    assert log == ["ocr", "close_ocr", "vision", "unload"]
    assert f.value("mileage") == 1234
