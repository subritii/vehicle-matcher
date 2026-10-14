"""Fake OCR engines for tests/test_ocr_timeout.py.

They live in their own module so the OCR worker process can import them."""

import time

from app.extract.invoice import Seg


class QuickOcr:
    """Returns the image width as text; sleeps 30 s on a 99-pixel-wide image."""

    name = "quick"
    word_level = False

    def __call__(self, image):
        if image.width == 99:
            time.sleep(30)
        return [Seg(f"W{image.width}", 0, 0, 10, 10, 0.9)]


class BrokenOcr:
    name = "broken"

    def __init__(self):
        raise ImportError("no such OCR library")
