"""Disk cache of raw engine outputs: OCR segments, fast-alpr reads, vision model answers
and vPIC decodes, keyed by image content and engine (model name and prompt text for the
vision model). The rules re-run on cached outputs, so changing a rule, a weight or the
matching never needs OCR or the model again. Wrappers keep their engine's interface.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from app.extract.invoice import Seg
from app.extract.photo import PlateRead
from app.extract.vin import VpicDecode


def image_key(image) -> str:
    h = hashlib.sha256(f"{image.mode}{image.size}".encode())
    h.update(image.tobytes())
    return h.hexdigest()[:32]


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


class DiskCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    def get(self, kind: str, key: str):
        p = self._path(kind, key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, kind: str, key: str, value) -> None:
        p = self._path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))


class CachedOcr:
    """``name`` is the engine the key is for (e.g. "paddle"), given up front so a cache
    hit never starts the OCR worker."""

    def __init__(self, inner, cache: DiskCache, name: str) -> None:
        self.inner, self.cache, self.name = inner, cache, name
        self.word_level = False

    def __call__(self, image) -> list[Seg]:
        key = _key(self.name, image_key(image))
        hit = self.cache.get("ocr", key)
        if hit is None:
            segs = self.inner(image)
            hit = {"word_level": bool(getattr(self.inner, "word_level", False)),
                   "segs": [dataclasses.astuple(s) for s in segs]}
            self.cache.put("ocr", key, hit)
        self.word_level = hit["word_level"]
        return [Seg(*s) for s in hit["segs"]]

    def close(self) -> None:
        self.inner.close()


class CachedAlpr:
    def __init__(self, inner, cache: DiskCache, name: str = "fast-alpr") -> None:
        self.inner, self.cache = inner, cache
        self.name = f"{name}:{getattr(inner, 'detector_model', '')}:{getattr(inner, 'ocr_model', '')}"

    def __call__(self, image) -> list[PlateRead]:
        key = _key(self.name, image_key(image))
        hit = self.cache.get("alpr", key)
        if hit is None:
            hit = [[r.text, r.confidence, list(r.box)] for r in self.inner(image)]
            self.cache.put("alpr", key, hit)
        return [PlateRead(t, c, tuple(b)) for t, c, b in hit]

    def close(self) -> None:
        self.inner.close()


class CachedVision:
    """Caches read_photo / read_invoice answers; errors are not cached."""

    def __init__(self, inner, cache: DiskCache) -> None:
        self.inner, self.cache = inner, cache
        self.model, self.name = inner.model, inner.name

    def _ask(self, prompt: str, images: list, call) -> dict:
        from app.extract.vision import load_prompt

        key = _key(self.model, prompt, load_prompt(prompt), *(image_key(im) for im in images))
        hit = self.cache.get("vision", key)
        if hit is None:
            hit = call(images)
            self.cache.put("vision", key, hit)
        return hit

    def read_photo(self, images: list) -> dict:
        return self._ask("photo_fields", images, self.inner.read_photo)

    def read_invoice(self, images: list) -> dict:
        return self._ask("invoice_fields", images, self.inner.read_invoice)

    def unload(self) -> None:
        self.inner.unload()

    def session(self, *args, **kwargs):
        return self.inner.session(*args, **kwargs)


def cached_vpic(cache: DiskCache, lookup=None):
    """VIN -> VpicDecode, remembering successful decodes on disk."""

    def vpic(vin: str) -> VpicDecode:
        hit = cache.get("vpic", vin)
        if hit is not None:
            return VpicDecode(**{**hit, "error_codes": tuple(hit["error_codes"])})
        from app.extract.vin import decode_vpic

        d = (lookup or decode_vpic)(vin)
        if d.ok:
            cache.put("vpic", vin, dataclasses.asdict(d))
        return d

    return vpic
