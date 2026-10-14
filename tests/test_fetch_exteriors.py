"""Offline checks for tools/fetch_exteriors.py (no network)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fetch_exteriors as fe  # noqa: E402


@pytest.mark.parametrize("model, title, hit", [
    ("CR-V", "File:'17-'19 Honda CR-V Paris.jpg", True),
    ("CR-V", "File:2018 Honda CRV EX.jpg", True),
    ("F-150", "File:2016 Ford F150 XLT.jpg", True),
    ("Silverado 1500", "File:2019 Chevrolet Silverado LT.jpg", True),
    ("1500", "File:2020 Ram 1500 Big Horn.jpg", True),
    ("1500", "File:2020 Ram 15000.jpg", False),
    ("Model 3", "File:Tesla Model 3 (2018).jpg", True),
    ("Civic", "File:Honda Civic Type R.jpg", True),
    ("Civic", "File:Honda Civica.jpg", False),
])
def test_model_pattern(model, title, hit):
    assert bool(fe.model_pattern(model).search(title)) == hit


@pytest.mark.parametrize("lic, ok", [
    ("CC BY-SA 4.0", True), ("CC BY 2.0", True), ("CC0", True), ("Public domain", True),
    ("GFDL", False), ("Copyrighted free use", False), ("All rights reserved", False),
])
def test_free_licenses_only(lic, ok):
    assert bool(fe.FREE_LICENSE.match(lic)) == ok


def test_titles_of_non_exteriors_are_skipped():
    assert fe.SKIP_TITLE.search("File:Honda Accord interior.jpg")
    assert fe.SKIP_TITLE.search("File:Ford F-150 Police Interceptor.jpg")
    assert not fe.SKIP_TITLE.search("File:2019 Honda Accord EX-L front.jpg")


def test_strip_html_author():
    assert fe.strip_html('<a href="//commons.wikimedia.org/wiki/User:X">Jane&nbsp;Doe</a>') == "Jane Doe"
