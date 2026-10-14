"""Per-image OCR timeout: OCR runs in a worker process that is killed when an image
takes too long; the image then counts as unreadable and the invoice goes to review."""

import time

import pytest
from PIL import Image

from app.extract import invoice as inv
from app.extract.invoice import OcrTimeout, TimedOcr

from tests.ocr_fakes import BrokenOcr, QuickOcr
from tests.test_invoice import BODY_SHOP, FakeOcr, Seg, _pdf


@pytest.fixture
def timed():
    ocr = TimedOcr(QuickOcr, timeout=3)
    yield ocr
    ocr.close()


def test_worker_returns_segments_and_engine_attributes(timed):
    segs = timed(Image.new("RGB", (120, 40), "white"))
    assert [s.text for s in segs] == ["W120"]
    assert timed.name == "quick" and timed.word_level is False


def test_slow_image_times_out_and_worker_recovers(timed):
    timed(Image.new("RGB", (50, 40), "white"))  # warm up the worker
    t = time.perf_counter()
    with pytest.raises(OcrTimeout):
        timed(Image.new("RGB", (99, 40), "white"))
    assert time.perf_counter() - t < 10
    assert [s.text for s in timed(Image.new("RGB", (77, 40), "white"))] == ["W77"]


def test_engine_that_cannot_load_raises():
    ocr = TimedOcr(BrokenOcr, timeout=3)
    try:
        with pytest.raises(RuntimeError, match="no such OCR library"):
            ocr.start()
    finally:
        ocr.close()


@pytest.mark.parametrize("raw, expected", [(None, 120.0), ("", 120.0), ("45", 45.0), ("0", 0.0)])
def test_ocr_timeout_setting(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("OCR_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("OCR_TIMEOUT", raw)
    assert inv.ocr_timeout() == expected


class TimeoutOcr:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        raise OcrTimeout("OCR took longer than 120 s")


def test_timed_out_image_is_unreadable_and_goes_to_review(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 100), "white").save(path)
    f = inv.extract_invoice(path, ocr=TimeoutOcr(), cache_dir=tmp_path / "cache", vpic=False)
    assert f.fields == {}
    assert f.needs_review
    assert any("OCR timed out" in r for r in f.review_reasons)
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())  # not cached


def test_pdf_ocr_fill_timeout_keeps_text_layer_and_goes_to_review(tmp_path):
    lines = [l for l in BODY_SHOP.splitlines() if not l.startswith("Odometer")] + ["Odometer:"]
    path = _pdf(tmp_path, lines)
    f = inv.extract_invoice(path, ocr=TimeoutOcr(), vpic=False)
    assert f.value("vin") is not None and f.fields["vin"].source == "pdf_text"
    assert f.value("mileage") is None
    assert any("OCR timed out" in r for r in f.review_reasons)
