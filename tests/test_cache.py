"""Disk cache of raw engine outputs: rules re-run without OCR, fast-alpr or the model."""

import contextlib

from PIL import Image

from app.extract import cache as ch
from app.extract.invoice import Seg
from app.extract.photo import PlateRead
from app.extract.vin import VpicDecode


def img(color="gray", size=(40, 30)):
    return Image.new("RGB", size, color)


class Ocr:
    name, word_level = "paddle", True

    def __init__(self):
        self.calls, self.closed = 0, 0

    def __call__(self, image):
        self.calls += 1
        return [Seg("VIN 123", 1, 2, 3, 4, 0.9)]

    def close(self):
        self.closed += 1


class Alpr:
    def __init__(self):
        self.calls, self.closed = 0, 0

    def __call__(self, image):
        self.calls += 1
        return [PlateRead("7KJD221", 0.9, (0, 0, 10, 5))]

    def close(self):
        self.closed += 1


class Vision:
    def __init__(self, model="m1"):
        self.model, self.name, self.calls, self.unloads, self.sessions = model, f"vision:{model}", [], 0, 0

    def read_photo(self, images):
        self.calls.append("photo")
        return {"shot_type": "plate"}

    def read_invoice(self, images):
        self.calls.append("invoice")
        return {"vin": None}

    def unload(self):
        self.unloads += 1

    @contextlib.contextmanager
    def session(self):
        self.sessions += 1
        yield self


def test_image_key_follows_content():
    assert ch.image_key(img()) == ch.image_key(img())
    assert ch.image_key(img()) != ch.image_key(img("red"))
    assert ch.image_key(img(size=(40, 30))) != ch.image_key(img(size=(30, 40)))


def test_ocr_is_cached_by_image_and_engine(tmp_path):
    inner = Ocr()
    ocr = ch.CachedOcr(inner, ch.DiskCache(tmp_path), name="paddle")
    assert ocr(img()) == ocr(img()) == [Seg("VIN 123", 1, 2, 3, 4, 0.9)]
    assert inner.calls == 1 and ocr.word_level and ocr.name == "paddle"
    again = ch.CachedOcr(Ocr(), ch.DiskCache(tmp_path), name="paddle")
    again(img())
    assert again.inner.calls == 0
    other = ch.CachedOcr(Ocr(), ch.DiskCache(tmp_path), name="tesseract")
    other(img())
    assert other.inner.calls == 1
    ocr.close()
    assert inner.closed == 1


def test_alpr_is_cached(tmp_path):
    inner = Alpr()
    alpr = ch.CachedAlpr(inner, ch.DiskCache(tmp_path))
    assert alpr(img()) == alpr(img()) == [PlateRead("7KJD221", 0.9, (0, 0, 10, 5))]
    assert inner.calls == 1
    assert ch.CachedAlpr(Alpr(), ch.DiskCache(tmp_path))(img("red"))  # new image: asks
    alpr.close()
    assert inner.closed == 1


def test_vision_is_cached_by_model_prompt_and_images(tmp_path):
    inner = Vision()
    v = ch.CachedVision(inner, ch.DiskCache(tmp_path))
    assert v.name == "vision:m1" and v.model == "m1"
    assert v.read_photo([img()]) == v.read_photo([img()]) == {"shot_type": "plate"}
    v.read_invoice([img()])
    assert inner.calls == ["photo", "invoice"]
    other = ch.CachedVision(Vision("m2"), ch.DiskCache(tmp_path))
    other.read_photo([img()])
    assert other.inner.calls == ["photo"]
    with v.session():
        pass
    v.unload()
    assert inner.sessions == 1 and inner.unloads == 1


def test_vision_errors_are_not_cached(tmp_path):
    from app.extract.vision import VisionError

    class Failing(Vision):
        def read_photo(self, images):
            self.calls.append("photo")
            raise VisionError("down")

    inner = Failing()
    v = ch.CachedVision(inner, ch.DiskCache(tmp_path))
    for _ in range(2):
        try:
            v.read_photo([img()])
        except VisionError:
            pass
    assert inner.calls == ["photo", "photo"]


def test_vpic_successes_are_cached(tmp_path):
    calls = []

    def lookup(vin):
        calls.append(vin)
        return VpicDecode(vin=vin, ok=vin != "BAD", make="HONDA", model="Accord", year=2003, error_codes=("0",))

    vpic = ch.cached_vpic(ch.DiskCache(tmp_path), lookup)
    assert vpic("V1") == vpic("V1") and vpic("V1").clean
    vpic("BAD"), vpic("BAD")
    assert calls == ["V1", "BAD", "BAD"]
    assert ch.cached_vpic(ch.DiskCache(tmp_path), lookup)("V1").make == "HONDA" and calls.count("V1") == 1
