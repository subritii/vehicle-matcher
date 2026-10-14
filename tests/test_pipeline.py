"""Field boxes on the invoice image and observation round trips."""

from datetime import date, datetime
from decimal import Decimal

from app import pipeline
from app.extract.invoice import Page
from app.match.cluster import PhotoObs
from app.match.score import InvoiceObs

TEXT = """\
NORTHWIND AUTO BODY                 INVOICE
VIN: 1HGCM82633A004352   Mileage: 60,513
Date                     Total
05/04/2026               $1,025.74"""


def test_value_box_on_its_evidence_line():
    page = Page.from_text(TEXT)
    line = page.lines[1]
    x0, y0, x1, y1 = pipeline.locate(page, line.text, 60513)
    assert 0.5 < x0 < x1 <= 1 and 0.2 < y0 < y1 < 0.5  # "60,513", right half of line 2


def test_value_under_its_label_and_formatted_values():
    page = Page.from_text(TEXT)
    label = page.lines[2].text
    assert pipeline.locate(page, label, date(2026, 5, 4))[1] > 0.7  # found on the line below
    assert pipeline.locate(page, label, Decimal("1025.74"))[0] > 0.4


def test_distinctive_value_anywhere_and_line_fallback():
    page = Page.from_text(TEXT)
    assert pipeline.locate(page, "1HGCM82633A004352", "1HGCM82633A004352")[0] < 0.3  # evidence is not a line
    assert pipeline.locate(page, "no such line", "2003") is None  # short values are not searched for
    x0, _, x1, _ = pipeline.locate(page, page.lines[0].text, "something else")
    assert x0 == 0 and x1 > 0.8  # the whole evidence line


def test_observations_round_trip_through_json():
    inv = InvoiceObs("i1", vin="V", vin_trusted=True, date=date(2026, 5, 4), mileage=10)
    assert pipeline.invoice_obs_from_json(pipeline.invoice_obs_to_json(inv)) == inv
    p = PhotoObs("7", uploader="ana", time=datetime(2026, 5, 4, 9), time_from_exif=True, gps=(1.5, 2.5), vin="V")
    assert pipeline.photo_obs_from_json(pipeline.photo_obs_to_json(p)) == p
