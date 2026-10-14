"""Photo extraction. OCR, fast-alpr, the vision model and vPIC are all faked; the fakes
log a timeline so tests can check that two heavy engines are never loaded together."""

from datetime import datetime

import numpy as np
import pytest
import zxingcpp
from PIL import Image

from app.extract import photo as ph
from app.extract.invoice import Seg
from tests.test_invoice import HONDA, FakeVpic, make_vin

FORD = make_vin("1FTEW1EP_JF654321")
BAD_CHECK = HONDA[:8] + ("0" if HONDA[8] != "0" else "1") + HONDA[9:]
UPLOAD = datetime(2026, 5, 2, 9, 30)


# --- Fakes ---------------------------------------------------------------------------
# Each test image has a unique width; the fakes look up their answer by width.


class Timeline:
    def __init__(self):
        self.loaded: set[str] = set()
        self.events: list[str] = []
        self.overlaps: list[set[str]] = []

    def load(self, name):
        self.loaded.add(name)
        if len(self.loaded) > 1:
            self.overlaps.append(set(self.loaded))
        self.events.append(f"{name}:run")

    def unload(self, name):
        if name in self.loaded:
            self.loaded.discard(name)
            self.events.append(f"{name}:stop")


class FakeOcr:
    name = "fake"
    word_level = False

    def __init__(self, tl, texts):
        self.tl, self.texts, self.calls = tl, texts, []

    def __call__(self, image):
        self.tl.load("ocr")
        self.calls.append(image.width)
        lines = self.texts.get(image.width, "").split("\n")
        return [Seg(t, 0, 20 * i, 10 * len(t), 20 * i + 15, 0.95) for i, t in enumerate(lines) if t]

    def close(self):
        self.tl.unload("ocr")


class FakeAlpr:
    def __init__(self, tl, reads):
        self.tl, self.reads, self.calls = tl, reads, []

    def __call__(self, image):
        self.tl.load("alpr")
        self.calls.append(image.width)
        return self.reads.get(image.width, [])

    def close(self):
        self.tl.unload("alpr")


class FakeVision:
    name = "vision:fake"

    def __init__(self, tl, answers):
        self.tl, self.answers, self.calls = tl, answers, []

    def read_photo(self, images):
        self.tl.load("vision")
        [im] = images
        self.calls.append(im.width)
        return dict(self.answers.get(im.width, {}))

    def unload(self):
        self.tl.unload("vision")

    def session(self):
        import contextlib

        @contextlib.contextmanager
        def cm():
            try:
                yield self
            finally:
                self.unload()

        return cm()


def image(tmp_path, width, name=None, exif=None, color="gray"):
    p = tmp_path / (name or f"p{width}.jpg")
    Image.new("RGB", (width, 300), color).save(p, "JPEG", **({"exif": exif} if exif is not None else {}))
    return p


def engines(ocr=None, alpr=None, vision=None):
    tl = Timeline()
    e = ph.Engines(ocr=FakeOcr(tl, ocr or {}), alpr=FakeAlpr(tl, alpr or {}),
                   vision=FakeVision(tl, vision) if vision is not None else None)
    return e, tl


def run(tmp_path, specs, vpic=False, uploaded_at=UPLOAD, **kw):
    """specs: list of widths (or paths). Returns results in order plus the engines."""
    paths = [s if not isinstance(s, int) else image(tmp_path, s) for s in specs]
    e, tl = engines(**kw)
    res = ph.extract_photos(paths, e, uploaded_at=uploaded_at, vpic=vpic)
    assert not tl.overlaps, f"engines loaded together: {tl.overlaps}"
    assert not tl.loaded, f"left loaded: {tl.loaded}"
    return res, e, tl


# --- EXIF ----------------------------------------------------------------------------


def exif_block(when="2026:04:30 08:15:00", gps=(37.7749, -122.4194), make="Apple", model="iPhone 13"):
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = make, model
    ex.get_ifd(0x8769)[0x9003] = when  # DateTimeOriginal

    def dms(v):
        v = abs(v)
        d, m = int(v), int((v % 1) * 60)
        return (d, m, round(((v % 1) * 60 - m) * 60, 2))

    if gps:
        g = ex.get_ifd(0x8825)
        g[1], g[2] = ("N" if gps[0] >= 0 else "S"), dms(gps[0])
        g[3], g[4] = ("E" if gps[1] >= 0 else "W"), dms(gps[1])
    return ex


def test_exif_capture_time_gps_and_device(tmp_path):
    p = image(tmp_path, 401, exif=exif_block())
    ex = ph.read_exif(p)
    assert ex.captured_at == datetime(2026, 4, 30, 8, 15)
    assert ex.gps == pytest.approx((37.7749, -122.4194), abs=1e-3)
    assert ex.device == "Apple iPhone 13"


def test_capture_time_from_exif_beats_upload_time(tmp_path):
    [r], _, _ = run(tmp_path, [image(tmp_path, 402, exif=exif_block())], vision={})
    assert r.value("captured_at") == datetime(2026, 4, 30, 8, 15)
    assert r.fields["captured_at"].source == "exif"
    assert r.value("gps") == pytest.approx((37.7749, -122.4194), abs=1e-3)
    assert r.value("device") == "Apple iPhone 13"


def test_upload_time_is_the_lower_weight_fallback(tmp_path):
    [r], _, _ = run(tmp_path, [403], vision={})
    fv = r.fields["captured_at"]
    assert (fv.value, fv.source) == (UPLOAD, "upload")
    assert fv.confidence < ph.EXIF_CONFIDENCE
    assert "gps" not in r.fields


def test_upload_time_per_photo(tmp_path):
    a, b = image(tmp_path, 404), image(tmp_path, 405)
    later = datetime(2026, 5, 3, 7, 0)
    ra, rb = run(tmp_path, [a, b], vision={}, uploaded_at={a: UPLOAD, b: later})[0]
    assert (ra.value("captured_at"), rb.value("captured_at")) == (UPLOAD, later)


# --- Shot type -----------------------------------------------------------------------


def test_shot_type_from_the_model_and_text_read_only_where_it_pays(tmp_path):
    vision = {
        501: {"shot_type": "exterior"}, 502: {"shot_type": "door_jamb"}, 503: {"shot_type": "odometer"},
        504: {"shot_type": "plate"}, 505: {"shot_type": "interior"}, 506: {"shot_type": "vin_plate"},
    }
    res, e, _ = run(tmp_path, [501, 502, 503, 504, 505, 506], vision=vision)
    assert [r.value("shot_type") for r in res] == [
        "exterior", "door_jamb", "odometer", "plate", "interior", "vin_plate"]
    assert all(r.fields["shot_type"].source == "vision:fake" for r in res)
    assert sorted(e.ocr.calls) == [502, 503, 505, 506]  # sticker, odometer, interior (a VIN plate?), VIN plate
    assert sorted(e.alpr.calls) == [501, 504]  # exterior, plate


@pytest.mark.parametrize("said, shot", [
    ("Door-jamb sticker", "door_jamb"), ("license plate", "plate"), ("VIN plate", "vin_plate"),
    ("dashboard odometer", "odometer"), ("car exterior", "exterior"), ("selfie", "other"), (None, "other"),
])
def test_model_shot_type_is_normalized(tmp_path, said, shot):
    [r], _, _ = run(tmp_path, [510], vision={510: {"shot_type": said}})
    assert r.value("shot_type") == shot


def test_without_the_model_shot_type_comes_from_rules(tmp_path):
    sticker = barcode_photo(tmp_path, HONDA, 1104)
    res, e, _ = run(tmp_path, [sticker, 521, 522, 523, 524], vision=None,
                    ocr={521: "ODO 60513 mi\nTRIP A 123.4", 522: f"VIN {HONDA}"},
                    alpr={523: [ph.PlateRead("7KJD221", 0.95, (100, 100, 400, 200))],  # fills the frame
                          524: [ph.PlateRead("7KJD221", 0.95, (10, 10, 40, 20))]})  # small: a car
    assert [r.value("shot_type") for r in res] == ["door_jamb", "odometer", "vin_plate", "plate", "exterior"]
    assert all(r.fields["shot_type"].source == "rules" for r in res)


def test_model_saying_other_does_not_hide_a_readable_vin(tmp_path):
    # The model called flat dashboard VIN plates "other"; OCR still looks, and a short text
    # with a VIN is a VIN plate or sticker. A document full of text (an invoice) stays "other".
    invoice = "\n".join([f"VIN {HONDA}"] + [f"line {i} parts and labor" for i in range(20)])
    vision = {525: {"shot_type": "other"}, 526: {"shot_type": "interior"}, 527: {"shot_type": "other"},
              528: {"shot_type": "other"}}
    res, e, _ = run(tmp_path, [525, 526, 527, 528], vision=vision,
                    ocr={525: HONDA, 526: f"MFD BY HONDA\nGVWR 2000 KG\n{FORD}", 527: invoice, 528: "EXIT"})
    assert [r.value("shot_type") for r in res] == ["vin_plate", "door_jamb", "other", "other"]
    assert [r.fields["shot_type"].source for r in res] == ["rules", "rules", "vision:fake", "vision:fake"]
    assert (res[0].value("vin"), res[1].value("vin")) == (HONDA, FORD)
    assert not res[0].vin_trusted and res[0].needs_review  # OCR alone: never trusted
    assert "vin" not in res[2].fields
    assert sorted(e.ocr.calls) == [525, 526, 527, 528]


# --- VIN -----------------------------------------------------------------------------


def barcode_photo(tmp_path, text, width=None):
    barcode = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.Code39)
    code = Image.fromarray(np.asarray(zxingcpp.write_barcode_to_image(barcode, scale=3)))
    canvas = Image.new("RGB", (width or 1100, 400), "white")
    canvas.paste(code.convert("RGB"), (100, 120))
    p = tmp_path / f"sticker{canvas.width}.jpg"
    canvas.save(p, "JPEG", quality=95)
    return p


def test_barcode_vin_is_trusted_and_skips_ocr(tmp_path):
    p = barcode_photo(tmp_path, HONDA, 1101)
    [r], e, _ = run(tmp_path, [p], vision={1101: {"shot_type": "door_jamb"}})
    assert r.value("vin") == HONDA and r.fields["vin"].source == "barcode"
    assert r.vin_trusted
    assert e.ocr.calls == []


def test_barcode_import_prefix_is_dropped(tmp_path):
    p = barcode_photo(tmp_path, "I" + HONDA, 1102)
    [r], _, _ = run(tmp_path, [p], vision={1102: {"shot_type": "door_jamb"}})
    assert r.value("vin") == HONDA


def test_barcode_failing_check_digit_falls_back_to_ocr(tmp_path):
    p = barcode_photo(tmp_path, BAD_CHECK, 1103)
    [r], e, _ = run(tmp_path, [p], vision={1103: {"shot_type": "door_jamb", "vin": HONDA}},
                    ocr={1103: f"VIN {HONDA}"})
    assert e.ocr.calls == [1103]
    assert r.value("vin") == HONDA and r.fields["vin"].source == "ocr:fake"


def test_ocr_vin_confirmed_by_the_model_is_trusted(tmp_path):
    [r], _, _ = run(tmp_path, [530], vision={530: {"shot_type": "vin_plate", "vin": HONDA.lower()}},
                    ocr={530: f"VIN {HONDA}"})
    assert r.value("vin") == HONDA and r.fields["vin"].source == "ocr:fake"
    assert r.vin_trusted
    assert ("vision", True) in [(c.source, c.agrees) for c in r.vin_confirmations]


def test_ocr_vin_alone_is_not_trusted(tmp_path):
    [r], _, _ = run(tmp_path, [531], vision={531: {"shot_type": "vin_plate"}}, ocr={531: f"VIN {HONDA}"})
    assert r.value("vin") == HONDA and not r.vin_trusted
    assert any("not confirmed" in x for x in r.review_reasons)


def test_ocr_and_model_reading_different_vins_blocks_trust(tmp_path):
    [r], _, _ = run(tmp_path, [532], vision={532: {"shot_type": "vin_plate", "vin": FORD}},
                    ocr={532: f"VIN {HONDA}"})
    assert r.value("vin") == HONDA and not r.vin_trusted
    assert any("vision model read a different VIN" in x for x in r.review_reasons)


def test_ocr_vin_failing_check_digit_is_reviewed_not_trusted(tmp_path):
    [r], _, _ = run(tmp_path, [533], vision={533: {"shot_type": "vin_plate", "vin": BAD_CHECK}},
                    ocr={533: f"VIN {BAD_CHECK}"})
    assert not r.vin_trusted


def test_model_only_vin_needs_a_clean_vpic_decode_and_is_never_trusted(tmp_path):
    vision = {534: {"shot_type": "vin_plate", "vin": HONDA}}
    [r], _, _ = run(tmp_path, [534], vision=vision, vpic=FakeVpic(year=2003, make="HONDA", model="Accord"))
    assert r.value("vin") == HONDA and r.fields["vin"].source == "vision:fake"
    assert not r.vin_trusted
    assert any("only by the vision model" in x for x in r.review_reasons)
    [r], _, _ = run(tmp_path, [534], vision=vision, vpic=False)
    assert "vin" not in r.fields
    [r], _, _ = run(tmp_path, [534], vision=vision, vpic=FakeVpic())  # vPIC request fails
    assert "vin" not in r.fields


def test_model_vin_failing_check_digit_is_ignored(tmp_path):
    [r], _, _ = run(tmp_path, [535], vision={535: {"shot_type": "vin_plate", "vin": BAD_CHECK}},
                    vpic=FakeVpic(year=2003, make="HONDA"))
    assert "vin" not in r.fields


def test_vin_not_read_from_photos_that_are_not_vin_shots(tmp_path):
    [r], e, _ = run(tmp_path, [536], vision={536: {"shot_type": "exterior", "vin": HONDA}},
                    vpic=FakeVpic(year=2003, make="HONDA"))
    assert "vin" not in r.fields


# --- Plate ---------------------------------------------------------------------------


def test_alpr_plate_confirmed_by_the_model_is_trusted_with_state(tmp_path):
    [r], _, _ = run(tmp_path, [540], vision={540: {"shot_type": "plate", "plate": "7kjd 221", "plate_state": "CA"}},
                    alpr={540: [ph.PlateRead("7KJD221", 0.7, (0, 0, 300, 100))]})
    assert (r.value("plate"), r.value("plate_state")) == ("7KJD221", "CA")
    assert r.fields["plate"].source == "alpr" and r.fields["plate_state"].source == "vision:fake"
    assert r.plate_trusted


def test_alpr_plate_alone_is_never_trusted(tmp_path):
    # Plates have no check digit: fast-alpr read "ODTR801" for "0DTR801" at 0.95 on dev photos.
    reads = {541: [ph.PlateRead("0DTR801", 1.0, (0, 0, 300, 100))]}
    [r], _, _ = run(tmp_path, [541], vision={541: {"shot_type": "plate"}}, alpr=reads)
    assert r.value("plate") == "0DTR801" and not r.plate_trusted
    assert any("plate" in x and "not confirmed" in x for x in r.review_reasons)
    [r], _, _ = run(tmp_path, [541], vision=None, alpr=reads)
    assert not r.plate_trusted


def test_plate_with_o_i_or_q_is_never_trusted(tmp_path):
    # Both readers read "OFYD327" for "0FYD327" on dev photos: agreement doesn't settle O/0, I/1, Q/0.
    for width, plate in ((546, "OFYD327"), (547, "7IKD221"), (548, "7QKD221")):
        [r], _, _ = run(tmp_path, [width], vision={width: {"shot_type": "plate", "plate": plate}},
                        alpr={width: [ph.PlateRead(plate, 0.99, (0, 0, 300, 100))]})
        assert r.value("plate") == plate and not r.plate_trusted
        assert any("O, I or Q" in x for x in r.review_reasons)


def test_plate_on_exterior_photo_is_never_trusted(tmp_path):
    # On exteriors both readers agreed on badges and dealer frames ("2009", "MODEL3").
    [r], _, _ = run(tmp_path, [549], vision={549: {"shot_type": "exterior", "plate": "MODEL3"}},
                    alpr={549: [ph.PlateRead("MODEL3", 0.9, (0, 0, 60, 20))]})
    assert r.value("plate") == "MODEL3" and not r.plate_trusted
    assert any("not a plate close-up" in x for x in r.review_reasons)


def test_biggest_plate_wins_on_exterior_photos(tmp_path):
    reads = {543: [ph.PlateRead("BGCAR01", 0.9, (0, 0, 30, 10)), ph.PlateRead("8BWD418", 0.9, (0, 0, 120, 40))]}
    [r], _, _ = run(tmp_path, [543], vision={543: {"shot_type": "exterior"}}, alpr=reads)
    assert r.value("plate") == "8BWD418"


def test_model_only_plate_is_never_trusted(tmp_path):
    [r], _, _ = run(tmp_path, [544], vision={544: {"shot_type": "plate", "plate": "7KJD221", "plate_state": "CA"}})
    assert r.value("plate") == "7KJD221" and r.fields["plate"].source == "vision:fake"
    assert not r.plate_trusted
    assert r.to_dict()["plate_trusted"] is False


def test_alpr_and_model_disagree_goes_to_review(tmp_path):
    [r], _, _ = run(tmp_path, [545], vision={545: {"shot_type": "plate", "plate": "7KJD227"}},
                    alpr={545: [ph.PlateRead("7KJD221", 0.97, (0, 0, 300, 100))]})
    assert r.value("plate") == "7KJD221" and not r.plate_trusted
    assert "plate" in r.conflicts and r.needs_review


# --- Odometer ------------------------------------------------------------------------


@pytest.mark.parametrize("text, miles", [
    ("ODO 60513 mi\nTRIP A 123.4", 60513),
    ("TRIP A 123.4\nTRIP B 9876.5\n60,513 mi", 60513),
    ("ODO\n060513", 60513),
    ("88 °F  D  ODO 101471 MILES", 101471),
    ("TRIP A 123.4", None),
    ("12:45 PM", None),
])
def test_parse_odometer(text, miles):
    assert ph.parse_odometer(text) == miles


def test_odometer_from_ocr_agrees_with_model(tmp_path):
    [r], _, _ = run(tmp_path, [550], vision={550: {"shot_type": "odometer", "odometer": "60,513"}},
                    ocr={550: "ODO 60513 mi"})
    assert r.value("odometer") == 60513 and r.fields["odometer"].source == "ocr:fake"
    assert not r.needs_review


def test_odometer_disagreement_goes_to_review(tmp_path):
    [r], _, _ = run(tmp_path, [551], vision={551: {"shot_type": "odometer", "odometer": "68513"}},
                    ocr={551: "ODO 60513 mi"})
    assert r.value("odometer") == 60513 and "odometer" in r.conflicts and r.needs_review


def test_model_fills_odometer_ocr_could_not_read(tmp_path):
    [r], _, _ = run(tmp_path, [552], vision={552: {"shot_type": "odometer", "odometer": "60513"}},
                    ocr={552: "ODO ---"})
    assert r.value("odometer") == 60513 and r.fields["odometer"].source == "vision:fake"


def test_implausible_odometer_from_model_is_dropped(tmp_path):
    [r], _, _ = run(tmp_path, [553], vision={553: {"shot_type": "odometer", "odometer": "1250000"}})
    assert "odometer" not in r.fields


# --- Color and body style ---------------------------------------------------------------


def test_color_and_body_style_from_exterior_photos(tmp_path):
    [r], _, _ = run(tmp_path, [560], vision={560: {"shot_type": "exterior", "color": "Pearl White",
                                                  "body_style": "Crossover SUV"}})
    assert (r.value("color"), r.value("body_style")) == ("white", "suv")
    assert r.fields["color"].source == r.fields["body_style"].source == "vision:fake"


@pytest.mark.parametrize("said, style", [
    ("4-door sedan", "sedan"), ("pickup truck", "pickup"), ("Truck", "pickup"), ("hatchback", "hatchback"),
    ("minivan", "minivan"), ("station wagon", "wagon"), ("spaceship", None),
])
def test_body_style_normalized(said, style):
    assert ph.normalize_body_style(said) == style


def test_color_ignored_on_close_ups(tmp_path):
    [r], _, _ = run(tmp_path, [561], vision={561: {"shot_type": "door_jamb", "color": "white", "body_style": "sedan"}})
    assert "color" not in r.fields and "body_style" not in r.fields


# --- Engines never overlap -----------------------------------------------------------


def test_engines_run_one_at_a_time_and_end_unloaded(tmp_path):
    vision = {570: {"shot_type": "vin_plate"}, 571: {"shot_type": "plate"}, 572: {"shot_type": "odometer"}}
    _, _, tl = run(tmp_path, [570, 571, 572], vision=vision, ocr={570: f"VIN {HONDA}", 572: "ODO 1234"},
                   alpr={571: [ph.PlateRead("7KJD221", 0.9, (0, 0, 300, 100))]})
    runs = [ev.split(":")[0] for ev in tl.events if ev.endswith(":run")]
    # one model pass, then OCR, then ALPR: each engine loads once for the whole batch
    assert runs == ["vision"] * 3 + ["ocr"] * 2 + ["alpr"]


def test_extract_photo_single(tmp_path):
    e, tl = engines(vision={580: {"shot_type": "odometer"}}, ocr={580: "ODO 4321 mi"})
    r = ph.extract_photo(image(tmp_path, 580), e, uploaded_at=UPLOAD, vpic=False)
    assert r.value("odometer") == 4321 and not tl.loaded and not tl.overlaps
    d = r.to_dict()
    assert d["fields"]["odometer"]["source"] == "ocr:fake" and d["fields"]["shot_type"]["value"] == "odometer"


def test_progress_reports_each_engine_pass_before_it_runs(tmp_path):
    e, tl = engines(vision={590: {"shot_type": "odometer"}, 591: {"shot_type": "plate"}}, ocr={590: "ODO 4321 mi"},
                    alpr={591: [ph.PlateRead("7KJD221", 0.9, (0, 0, 300, 100))]})
    seen = []
    ph.extract_photos([image(tmp_path, 590), image(tmp_path, 591)], e, uploaded_at=UPLOAD, vpic=False,
                      progress=lambda step: seen.append((step, len(tl.events))))
    assert [s for s, _ in seen] == ["vision", "ocr", "alpr"]
    assert seen[0][1] == 0  # before the model ran
