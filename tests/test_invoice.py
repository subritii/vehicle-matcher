"""Tests for app/extract/invoice.py.

Invoices here are hand-written in the styles of real shop paperwork (body shop
invoice, dealer repair order with boxed fields, quick-lube receipt). Page.from_text
turns plain text into positioned segments: runs of 2+ spaces split segments and
character columns become x positions, so "value under its label" layouts work.
"""

import math
from datetime import date
from decimal import Decimal

import pytest
from fpdf import FPDF
from PIL import Image

from app.extract import invoice as inv
from app.extract.invoice import Page, Seg, extract_from_page, read_page
from app.extract.vin import VpicDecode, compute_check_digit


def make_vin(template: str) -> str:
    """Fill the '_' at position 9 with the correct check digit."""
    assert len(template) == 17 and template[8] == "_"
    probe = template.replace("_", "0")
    return probe[:8] + compute_check_digit(probe) + probe[9:]


HONDA = "1HGCM82633A004352"  # 2003 Honda
HONDA_07 = "JHMCM56557C404453"  # 2007 Honda
CHEVY = make_vin("1GCUYDED_KZ123456")  # 2019 Chevrolet truck
FORD = make_vin("1FTEW1EP_JF654321")  # 2018 Ford


BODY_SHOP = f"""\
NORTHWIND AUTO BODY
1200 Harbor Way, Portland, OR 97201   (503) 555-0188
INVOICE
Invoice No: 2025-00417            Invoice Date: 03/14/2025
Bill To: Dana Whitfield           Claim #: 88-4410-2291
Vehicle: 2003 Honda Accord EX     Color: Silver Metallic
VIN: {HONDA}            License: 7KJD221 CA
Odometer: 84,211 mi
Description                       Qty      Amount
Front bumper cover                1        310.00
Refinish labor 2.5 hrs            1        180.00
Subtotal                                   490.00
Sales Tax                                   24.50
Total Due                                 $514.50
Warranty: 12 months / 12,000 miles
"""

DEALER_RO = f"""\
SUMMIT HONDA SERVICE CENTER
REPAIR ORDER  RO# 558213
DATE IN      DATE OUT     ADVISOR
04/02/2025   04/03/2025   K. SMITH
YEAR/MAKE/MODEL               V.I.N.
2007 HONDA ACCORD SE          {HONDA_07}
LIC PLATE    ODO IN    ODO OUT    COLOR
TX GHT4412   61022     61080      BLK
OP   DESCRIPTION              TOTAL
A    LOF SYNTHETIC            89.95
B    ROTATE TIRES             29.95
PARTS 0.00   LABOR 119.90   TAX 0.00
TOTAL CHARGES         $1,021.66
"""

RECEIPT = f"""\
QUICK LANE TIRE & AUTO
Receipt #: 0091827
Date: 2025-07-19
2019 Chevy Silverado 1500 LT
Plate: 8ABC123 / CA
Mileage In: 45,210   Mileage Out: 45,222
VIN
{CHEVY}
SUBTOTAL 100.00
TAX 8.00
TOTAL 108.00
Next service due at 50,210 mi
"""


def fields(text):
    return extract_from_page(Page.from_text(text))


# --- Whole invoices ---------------------------------------------------------------


def test_body_shop_invoice():
    f = fields(BODY_SHOP)
    assert f.value("vendor") == "NORTHWIND AUTO BODY"
    assert f.value("invoice_no") == "2025-00417"
    assert f.value("invoice_date") == date(2025, 3, 14)
    assert f.value("vin") == HONDA
    assert (f.value("plate"), f.value("plate_state")) == ("7KJD221", "CA")
    assert (f.value("year"), f.value("make"), f.value("model")) == (2003, "Honda", "Accord")
    assert f.value("color") == "silver"
    assert f.value("mileage") == 84211
    assert f.value("total") == Decimal("514.50")
    assert not f.needs_review, f.review_reasons


def test_dealer_repair_order_with_boxed_fields():
    f = fields(DEALER_RO)
    assert f.value("vendor") == "SUMMIT HONDA SERVICE CENTER"
    assert f.value("invoice_no") == "558213"
    assert f.value("invoice_date") == date(2025, 4, 3)  # date out, not date in
    assert f.value("vin") == HONDA_07
    assert (f.value("plate"), f.value("plate_state")) == ("GHT4412", "TX")
    assert (f.value("year"), f.value("make"), f.value("model")) == (2007, "Honda", "Accord")
    assert f.value("mileage") == 61022  # odometer in
    assert f.value("color") == "black"
    assert f.value("total") == Decimal("1021.66")
    assert not f.needs_review, f.review_reasons


def test_quick_lube_receipt():
    f = fields(RECEIPT)
    assert f.value("vendor") == "QUICK LANE TIRE & AUTO"
    assert f.value("invoice_no") == "0091827"
    assert f.value("invoice_date") == date(2025, 7, 19)
    assert f.value("vin") == CHEVY
    assert (f.value("plate"), f.value("plate_state")) == ("8ABC123", "CA")
    assert (f.value("year"), f.value("make"), f.value("model")) == (2019, "Chevrolet", "Silverado 1500")
    assert f.value("mileage") == 45210
    assert f.value("color") is None
    assert f.value("total") == Decimal("108.00")


def test_every_field_reports_confidence_in_range():
    f = fields(BODY_SHOP)
    for name in inv.FIELD_NAMES:
        fv = f.fields.get(name)
        if fv is not None:
            assert 0.0 < fv.confidence <= 1.0, name


# --- VIN handling and review ----------------------------------------------------


def test_two_trusted_vins_send_invoice_to_review():
    f = fields(f"ACME AUTO\nVIN: {HONDA}\nPrevious vehicle VIN: {HONDA_07}\nTotal 10.00\n")
    assert f.needs_review
    assert any("multiple VINs" in r for r in f.review_reasons)
    assert f.value("vin") in (HONDA, HONDA_07)


def test_same_vin_twice_is_not_multiple():
    f = fields(f"ACME AUTO\nVIN: {HONDA}\nVehicle {HONDA} 2003 Honda Accord\n")
    assert not any("multiple VINs" in r for r in f.review_reasons)


def test_unlabeled_assembled_vin_is_kept_low_confidence_and_reviewed():
    f = fields(f"ACME AUTO\nNotes: {HONDA[:8]} {HONDA[8:]}\n2003 Honda Accord\nTotal 10.00\n")
    assert f.value("vin") == HONDA
    assert f.fields["vin"].confidence <= 0.2
    assert f.needs_review


def test_untrusted_vin_does_not_count_against_a_trusted_one():
    # Unrelated short codes on one line can join into a check-digit-passing string.
    # (A line in between, so the VIN label isn't on the line above it.)
    text = f"ACME AUTO\nVIN: {HONDA}\nCustomer since 2019\nRef {HONDA_07[:8]} {HONDA_07[8:]}\n"
    f = fields(text)
    assert any(r.vin == HONDA_07 and r.unlabeled_assembly for r in f.vins)  # it was found...
    assert f.value("vin") == HONDA
    assert not any("multiple VINs" in r for r in f.review_reasons)  # ...but doesn't count


def test_vin_failing_check_digit_goes_to_review():
    f = fields("ACME AUTO\nVIN: 1HGCM82633A004353\n2003 Honda Accord\n")
    assert f.value("vin") == "1HGCM82633A004353"
    assert f.needs_review


def test_no_vin_and_no_plate_goes_to_review():
    f = fields("ACME AUTO\n2003 Honda Accord\nTotal 10.00\n")
    assert f.value("vin") is None
    assert f.needs_review


def test_plate_only_invoice_is_fine():
    f = fields("ACME AUTO\nDate: 01/02/2025\nPlate: ABC1234 NY\n2015 Toyota Camry\nTotal 10.00\n")
    assert f.value("vin") is None
    assert f.value("plate") == "ABC1234"
    assert not any("no VIN or plate" in r for r in f.review_reasons)


def test_year_and_make_fall_back_to_vin():
    f = fields(f"ACME AUTO\nVIN: {FORD}\nTotal 10.00\n")
    assert f.value("year") == 2018
    assert f.value("make") == "Ford"
    assert f.fields["year"].confidence < 0.9


def test_year_disagreeing_with_vin_goes_to_review():
    f = fields(f"ACME AUTO\nVIN: {FORD}\n2016 Ford F-150 XLT\n")
    assert f.value("year") == 2016  # what the invoice says
    assert any("year" in r and "VIN" in r for r in f.review_reasons)


# --- Plate ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, plate, state",
    [
        ("License: 7KJD221 CA", "7KJD221", "CA"),
        ("Lic Plate: TX GHT4412", "GHT4412", "TX"),
        ("Plate/State: ABC1234/NY", "ABC1234", "NY"),
        ("PLATE  KE36671 IL", "KE36671", "IL"),
        ("Plate: XVB9960 (WA)", "XVB9960", "WA"),
        ("Lic#: ABC 1234 FL", "ABC1234", "FL"),
        ("Registration: 4GDR772", "4GDR772", None),
        ("License Plate No. 8ABC123", "8ABC123", None),
    ],
)
def test_plate_formats(line, plate, state):
    f = fields(f"ACME AUTO\n{line}\n")
    assert (f.value("plate"), f.value("plate_state")) == (plate, state)


@pytest.mark.parametrize(
    "line, state",
    [("PLATE  KE36671 11", "IL"), ("PLATE  UBA2390 0H", "OH"), ("PLATE  ABC1234 12", None)],
)
def test_plate_state_with_ocr_digit_for_letter(line, state):
    f = fields(f"ACME AUTO\n{line}\n")
    assert f.value("plate_state") == state
    if state:
        assert f.fields["plate_state"].confidence < f.fields["plate"].confidence


def test_tag_hat_number_is_not_a_plate():
    f = fields("ACME AUTO\nTAG / HAT    MILEAGE IN\nT699         39916\n")
    assert f.value("plate") is None


# --- Vehicle ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Vehicle: 2019 Chevy Silverado 1500 LT", (2019, "Chevrolet", "Silverado 1500")),
        ("2016 JEEP GRAND CHEROKEE LIMITED", (2016, "Jeep", "Grand Cherokee")),
        ("Ford F150 XLT 2018", (2018, "Ford", "F-150")),
        ("2021 Mercedes Benz C300 4MATIC", (2021, "Mercedes-Benz", "C300")),
        ("2020 Tesla Model Y Long Range", (2020, "Tesla", "Model Y")),
        ("2022 Ram 1500 Big Horn", (2022, "Ram", "1500")),
        ("2017 VW Tiguan SE", (2017, "Volkswagen", "Tiguan")),
        ("Year: 2017   Make: Toyota   Model: RAV4", (2017, "Toyota", "RAV4")),
        ("Make: Honda   Model: CR-V   Year: 2019", (2019, "Honda", "CR-V")),
        ("Kia Telluride (2023)", (2023, "Kia", "Telluride")),
        ("2020 BMW 3301", (2020, "BMW", "330i")),  # OCR: i read as 1
        ("2018 Mazda CX-S", (2018, "Mazda", "CX-5")),  # OCR: 5 read as S
    ],
)
def test_vehicle_descriptions(line, expected):
    f = fields(f"ACME AUTO\n{line}\n")
    assert (f.value("year"), f.value("make"), f.value("model")) == expected


def test_dealer_name_with_make_is_not_the_vehicle():
    f = fields("SMITH TOYOTA OF SPRINGFIELD\nInvoice 10023\n2018 Honda Civic LX\n")
    assert f.value("make") == "Honda"
    assert f.value("model") == "Civic"


def test_unknown_model_after_make_is_low_confidence():
    f = fields("ACME AUTO\n2019 Honda Zephyrix\n")
    assert f.value("make") == "Honda"
    assert f.value("model") == "Zephyrix"
    assert f.fields["model"].confidence < 0.6


def test_year_not_taken_from_a_date():
    f = fields("ACME AUTO\nDate: 03/14/2025\nHonda Accord\n")
    assert f.value("year") is None


# --- Mileage ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Odometer: 84,211 mi", 84211),
        ("Mileage In: 45,210   Mileage Out: 45,222", 45210),
        ("Mileage Out: 45,222   Mileage In: 45,210", 45210),
        ("ODO IN/OUT 33100/33104", 33100),
        ("Miles: 120500", 120500),
        ("Odometer reading 7,402", 7402),
        ("MILEAGE\n102810", 102810),
        ("Mileage: 99.306", 99306),  # OCR read the thousands comma as a dot
        ("000  27325", 27325),  # OCR read the label ODO as 000
        ("0D0  27340", 27340),
    ],
)
def test_mileage(text, expected):
    assert fields(f"ACME AUTO\n{text}\n").value("mileage") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Warranty 12 months / 12,000 miles",
        "Next service due at 50,210 mi",
        "Oil change 5,000 mile interval",
    ],
)
def test_mileage_ignores_warranty_and_service_text(text):
    assert fields(f"ACME AUTO\n{text}\n").value("mileage") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Mileage: 45210mi", 45210),
        ("Odometer (mi): 45,210", 45210),
        ("Mileage: 45,210, Color: Red", 45210),
    ],
)
def test_mileage_units_and_qualifiers(text, expected):
    assert fields(f"ACME AUTO\n{text}\n").value("mileage") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Mileage:   Plate: XVB9960",  # blank value: never borrow the next field's digits
        "Odometer:   Year: 2019",
        "Mileage: ____   Invoice #: 48213",
        "Mileage: XVB9960",  # digits glued to letters
        "Mileage: 45210A",
        "Mileage: A45210",
        "Odometer: 4S,210",  # OCR misread: not clean, leave blank
        "Mileage: 45,2l0",
        "Mileage: 45 210",  # two numbers: ambiguous
        "Mileage: 03/14/2024",
        "Mileage: $45.10",
    ],
)
def test_mileage_left_blank_when_not_clean(text):
    assert fields(f"ACME AUTO\n{text}\n").value("mileage") is None


def test_mileage_blank_does_not_take_plate_from_next_field():
    f = fields("ACME AUTO\nMileage:      Plate: XVB9960 CA\n")
    assert f.value("mileage") is None
    assert f.value("plate") == "XVB9960"


def test_mileage_under_label_must_be_clean():
    # Form layout: the box under "Mileage" holds a plate-like string, not a mileage.
    f = fields("ACME AUTO\nMileage      Color\nXVB9960      Red\n")
    assert f.value("mileage") is None
    f = fields("ACME AUTO\nMileage      Color\n45,210       Red\n")
    assert f.value("mileage") == 45210


def test_mileage_ocr_segments_with_blank_value():
    # OCR of a handwritten value that came out empty: next segment is another field.
    page = Page.from_segs(
        [Seg("ACME AUTO", 0, 0, 100, 20), Seg("Mileage:", 0, 40, 90, 60), Seg("Plate: XVB9960", 300, 40, 450, 60)],
        width=600, height=800, source="ocr:test",
    )
    assert extract_from_page(page).value("mileage") is None


# --- Dates ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s, expected",
    [
        ("03/14/2025", date(2025, 3, 14)),
        ("3/4/25", date(2025, 3, 4)),
        ("2025-07-19", date(2025, 7, 19)),
        ("14 March 2025", date(2025, 3, 14)),
        ("Mar 14, 2025", date(2025, 3, 14)),
        ("March 14 2025", date(2025, 3, 14)),
        ("14-Mar-2025", date(2025, 3, 14)),
        ("Sept. 2, 2025", date(2025, 9, 2)),
        ("25/12/2025", date(2025, 12, 25)),  # day > 12, so day-first
        ("13/13/2025", None),
        ("no date here", None),
    ],
)
def test_parse_date(s, expected):
    assert inv.parse_date(s) == expected


def test_invoice_date_preferred_over_other_dates():
    text = "ACME AUTO\nDate In: 01/02/2025\nInvoice Date: 01/05/2025\nDue Date: 02/04/2025\n"
    assert fields(text).value("invoice_date") == date(2025, 1, 5)


def test_due_date_never_used():
    assert fields("ACME AUTO\nDue Date: 02/04/2025\n").value("invoice_date") is None


def test_unlabeled_date_near_top_is_a_fallback():
    f = fields("ACME AUTO\n02/11/2025\nSomething\n")
    assert f.value("invoice_date") == date(2025, 2, 11)
    assert f.fields["invoice_date"].confidence < 0.9


# --- Total ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Subtotal 400.00\nTax 33.00\nTotal 433.00", "433.00"),
        ("AMOUNT DUE $1,204.33", "1204.33"),
        ("Total 433.00\nBalance Due 233.00", "233.00"),  # payment applied
        ("Total Parts 120.00\nTotal Labor 200.00\nGrand Total 345.60", "345.60"),
        ("Item          Total\nBrake pads    120.00\nTotal         120.00", "120.00"),
        ("TOTAL\n$88.10", "88.10"),
        ("Sub Total 50.00\nTotal Tax 4.00", None),
    ],
)
def test_total(text, expected):
    got = fields(f"ACME AUTO\n{text}\n").value("total")
    assert got == (Decimal(expected) if expected else None)


# --- Invoice number ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Invoice #: A-10023", "A-10023"),
        ("INVOICE NO. 55821", "55821"),
        ("RO# 123456", "123456"),
        ("Repair Order: 448812", "448812"),
        ("Work Order WO-7781", "WO-7781"),
        ("Receipt 0091827", "0091827"),
        ("INVOICE\nNumber: 7781-B", "7781-B"),
    ],
)
def test_invoice_no(text, expected):
    assert fields(f"ACME AUTO\n{text}\n").value("invoice_no") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Invoice Date: 03/14/2025",
        "R.O. DATE 2025-03-14",
        "Phone (503) 555-0188",
    ],
)
def test_invoice_no_never_a_date_or_phone(text):
    assert fields(f"ACME AUTO\n{text}\n").value("invoice_no") is None


# --- Color, vendor ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Silver Metallic", "silver"), ("SIL", "silver"), ("Pearl White", "white"), ("WHT", "white"),
        ("BLK", "black"), ("Magnetic Gray", "gray"), ("grey", "gray"), ("GRY", "gray"),
        ("Deep Blue Pearl", "blue"), ("Maroon", "red"), ("Burgundy", "red"), ("Tan", "beige"),
        ("Pearl", None), ("NH-731P", None),
    ],
)
def test_normalize_color(raw, expected):
    assert inv.normalize_color(raw) == expected


def test_paint_line_item_is_not_a_color():
    assert fields("ACME AUTO\nPaint materials 98.84\nRefinish labor 2.0\n").value("color") is None


def test_vendor_prefers_largest_text_near_top():
    page = Page.from_segs(
        [
            Seg("INVOICE", 400, 10, 500, 30),
            Seg("Riverbend Collision", 20, 12, 220, 40),  # tallest
            Seg("12 Mill Rd, Fairview TX", 20, 45, 200, 55),
            Seg("Invoice #: 7781", 400, 40, 500, 50),
        ],
        width=600, height=800, source="test",
    )
    assert extract_from_page(page).value("vendor") == "Riverbend Collision"


@pytest.mark.parametrize("skip", ["INVOICE", "REPAIR ORDER", "Final Invoice", "1200 Harbor Way", "(503) 555-0188"])
def test_vendor_skips_document_titles_addresses_and_phones(skip):
    f = fields(f"{skip}\nACME AUTO REPAIR\n")
    assert f.value("vendor") == "ACME AUTO REPAIR"


# --- Layout: OCR line grouping ----------------------------------------------------


def _rotated(text, x, y, w, h, deg):
    """A segment whose baseline is rotated by deg around (0, 0)."""
    a = math.radians(deg)
    cx, cy = x + w / 2, y + h / 2
    rx, ry = cx * math.cos(a) - cy * math.sin(a), cx * math.sin(a) + cy * math.cos(a)
    return Seg(text, rx - w / 2, ry - h / 2, rx + w / 2, ry + h / 2, score=0.99, angle=deg)


def test_skewed_ocr_segments_group_into_lines():
    # Two rows of label/value pairs on a page photographed at 4 degrees.
    segs = [
        _rotated("VIN:", 50, 100, 60, 20, 4), _rotated(HONDA, 120, 100, 300, 20, 4),
        _rotated("Mileage:", 700, 100, 120, 20, 4), _rotated("84,211", 830, 100, 90, 20, 4),
        _rotated("Plate:", 50, 140, 80, 20, 4), _rotated("7KJD221 CA", 140, 140, 160, 20, 4),
    ]
    page = Page.from_segs(segs, width=1200, height=1600, source="ocr:test")
    assert [line.text for line in page.lines] == [
        f"VIN:  {HONDA}  Mileage:  84,211",
        "Plate:  7KJD221 CA",
    ]
    f = extract_from_page(page)
    assert f.value("mileage") == 84211
    assert f.value("plate") == "7KJD221"


def test_keystone_skew_groups_rows_by_local_angle():
    # Phone photo with perspective: the text angle grows from 2 to 4 degrees down the
    # page, so one global deskew angle leaves right-hand values a row off.
    segs = []
    labels = ["Subtotal", "Tax", "Total", "Paid", "Balance", "Change"]
    for i, label in enumerate(labels):
        a = math.radians(2 + 0.4 * i)
        y = 100 + 30 * i
        segs.append(Seg(f"{label} amount", 50, y, 250, y + 20, angle=math.degrees(a)))
        vy = y + math.tan(a) * (920 - 150)  # value center follows the label center's baseline
        segs.append(Seg(f"{i + 1}.00", 880, vy, 960, vy + 20, angle=math.degrees(a)))
    page = Page.from_segs(segs, width=1000, height=1400, source="ocr:test")
    assert [line.text for line in page.lines] == [f"{lab} amount  {i + 1}.00" for i, lab in enumerate(labels)]


def test_field_confidence_scaled_by_ocr_score():
    page = Page.from_segs(
        [Seg("ACME AUTO", 0, 0, 100, 20, score=0.99), Seg("Odometer: 84,211", 0, 40, 200, 60, score=0.5)],
        width=600, height=800, source="ocr:test",
    )
    assert extract_from_page(page).fields["mileage"].confidence <= 0.5


# --- Reading files ----------------------------------------------------------------


class FakeOcr:
    name = "fake"

    def __init__(self, segs):
        self.segs = segs
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return self.segs


def _pdf(tmp_path, lines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    path = tmp_path / "doc.pdf"
    pdf.output(str(path))
    return path


def test_read_pdf_uses_text_layer_without_ocr(tmp_path):
    path = _pdf(tmp_path, ["ACME AUTO", f"VIN: {HONDA}", "Total Due $10.00"])
    ocr = FakeOcr([])
    page = read_page(path, ocr=ocr)
    assert page.source == "pdf_text"
    assert ocr.calls == 0
    assert extract_from_page(page).value("vin") == HONDA


def test_read_pdf_without_text_layer_falls_back_to_ocr(tmp_path):
    pdf = FPDF()
    pdf.add_page()  # blank page: no text layer
    path = tmp_path / "scan.pdf"
    pdf.output(str(path))
    ocr = FakeOcr([Seg("ACME AUTO", 10, 10, 200, 40), Seg(f"VIN: {HONDA}", 10, 60, 400, 80)])
    page = read_page(path, ocr=ocr)
    assert ocr.calls == 1
    assert page.source == "ocr:fake"
    assert extract_from_page(page).value("vin") == HONDA


def test_read_image_uses_ocr(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 300), "white").save(path)
    ocr = FakeOcr([Seg(f"VIN {HONDA}", 10, 10, 190, 30)])
    page = read_page(path, ocr=ocr)
    assert page.source == "ocr:fake"
    assert page.width == 200 and page.height == 300


def test_ocr_cache_avoids_second_call(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (50, 50), "white").save(path)
    ocr = FakeOcr([Seg("ACME AUTO", 1, 1, 40, 10, score=0.9, angle=1.5)])
    first = read_page(path, ocr=ocr, cache_dir=tmp_path / "cache")
    second = read_page(path, ocr=ocr, cache_dir=tmp_path / "cache")
    assert ocr.calls == 1
    assert second.lines[0].segs == first.lines[0].segs


def test_extract_invoice_end_to_end_on_pdf(tmp_path):
    path = _pdf(tmp_path, BODY_SHOP.splitlines())
    ocr = FakeOcr([])
    f = inv.extract_invoice(path, ocr=ocr, vpic=False)
    assert f.text_source == "pdf_text"
    assert f.value("vin") == HONDA
    assert f.value("total") == Decimal("514.50")
    assert ocr.calls == 0  # every field was in the text layer
    assert f.vin_trusted


# --- Review: repaired VINs and unknown manufacturer codes ---------------------------


def test_repaired_vin_goes_to_review():
    f = fields("ACME AUTO\nVIN: 1HGCM82633A0043S2\n2003 Honda Accord\n")  # S read for 5
    assert f.value("vin") == HONDA
    assert any("repaired" in r for r in f.review_reasons)
    assert not f.vin_trusted


def test_unknown_wmi_goes_to_review():
    odd = make_vin("9ZZCM826_3A004352")
    f = fields(f"ACME AUTO\nVIN: {odd}\n2003 Honda Accord\n")
    assert f.value("vin") == odd
    assert any("WMI" in r and "9ZZ" in r for r in f.review_reasons)
    assert not f.vin_trusted


@pytest.mark.parametrize("vin_", [HONDA, HONDA_07, CHEVY, FORD, make_vin("1C4RJFAG_HC637453")])
def test_known_wmis_are_not_flagged(vin_):
    f = fields(f"ACME AUTO\nVIN: {vin_}\n")
    assert not any("WMI" in r for r in f.review_reasons)


def test_shared_wmi_gives_no_make_fallback():
    # 1C4 is used by Chrysler, Dodge and Jeep: known WMI, but it doesn't name one make.
    f = fields(f"ACME AUTO\nVIN: {make_vin('1C4RJFAG_HC637453')}\n")
    assert f.value("make") is None
    assert f.value("year") == 2017


# --- VIN trust: text layer vs OCR ---------------------------------------------------


class FakeVpic:
    def __init__(self, **decoded):
        self.decoded = decoded
        self.calls = []

    def __call__(self, vin_):
        self.calls.append(vin_)
        if not self.decoded:
            return VpicDecode(vin=vin_, ok=False, error_text="vPIC request failed: timeout")
        return VpicDecode(vin=vin_, **{"ok": True, "error_codes": ("0",), **self.decoded})


def ocr_page(text):
    page = Page.from_text(text)
    page.source = "ocr:test"
    return page


OCR_INVOICE = f"ACME AUTO\nVIN: {HONDA}\n2003 Honda Accord EX\nPlate: 7KJD221 CA\n"


def test_text_layer_vin_is_trusted_without_vpic():
    vpic = FakeVpic()
    f = extract_from_page(Page.from_text(OCR_INVOICE), vpic=vpic)
    assert f.vin_trusted
    assert vpic.calls == []
    assert [c.source for c in f.vin_confirmations] == ["text_layer"]


def test_ocr_vin_trusted_when_vpic_agrees_with_printed_year_and_make():
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f = extract_from_page(ocr_page(OCR_INVOICE), vpic=vpic)
    assert vpic.calls == [HONDA]
    assert f.vin_trusted
    assert not f.needs_review, f.review_reasons
    [c] = f.vin_confirmations
    assert (c.source, c.agrees) == ("vpic", True)


@pytest.mark.parametrize(
    "decoded, why",
    [
        ({"year": 2004, "make": "HONDA"}, "year"),
        ({"year": 2003, "make": "ACURA"}, "make"),
        ({"year": 2003, "make": "HONDA", "model": "Civic"}, "model"),
        ({"year": 2003, "make": "HONDA", "error_codes": ("5",)}, "errors in few positions"),
        ({"year": 2003, "make": "HONDA", "error_codes": ("0", "1")}, "check digit"),
        ({"year": 2003, "make": "HONDA", "error_codes": ("14",)}, "partial, no model to compare"),
        ({"year": 2003, "make": "HONDA", "model": "Civic", "error_codes": ("14",)}, "partial, model differs"),
        ({}, "failed"),
    ],
)
def test_ocr_vin_not_trusted_when_vpic_disagrees_or_fails(decoded, why):
    f = extract_from_page(ocr_page(OCR_INVOICE), vpic=FakeVpic(**decoded))
    assert not f.vin_trusted
    assert any("not confirmed" in r for r in f.review_reasons)
    assert f.value("vin") == HONDA  # the value is still reported


@pytest.mark.parametrize("codes", [("8",), ("7",), ("0", "12"), ("14",)])
def test_ocr_vin_trusted_with_informational_vpic_codes(codes):
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord", error_codes=codes)
    f = extract_from_page(ocr_page(OCR_INVOICE), vpic=vpic)
    assert f.vin_trusted, f.review_reasons


def test_ocr_vin_needs_year_and_make_printed_on_the_invoice():
    # Year/make derived from the VIN itself can't confirm the VIN.
    vpic = FakeVpic(year=2003, make="HONDA")
    f = extract_from_page(ocr_page(f"ACME AUTO\nVIN: {HONDA}\nPlate: 7KJD221 CA\n"), vpic=vpic)
    assert f.value("year") == 2003 and f.fields["year"].source == "vin"
    assert not f.vin_trusted
    assert any("not confirmed" in r for r in f.review_reasons)


def test_ocr_vin_with_vpic_disabled_goes_to_review():
    f = extract_from_page(ocr_page(OCR_INVOICE), vpic=False)
    assert not f.vin_trusted
    assert any("not confirmed" in r for r in f.review_reasons)


def test_more_confirmation_sources_can_be_added():
    f = extract_from_page(ocr_page(OCR_INVOICE), vpic=False)
    assert not f.vin_trusted
    f.add_vin_confirmation(inv.VinConfirmation("photo_vin", True, "matches door-jamb sticker"))
    assert f.vin_trusted
    assert not any("not confirmed" in r for r in f.review_reasons)


def test_unverified_vin_never_trusted_even_if_vpic_agrees():
    f = extract_from_page(ocr_page("ACME AUTO\nVIN: 1HGCM82633A004353\n2003 Honda Accord\n"),
                          vpic=FakeVpic(year=2003, make="HONDA"))
    assert not f.vin_trusted


def test_field_source_is_recorded():
    f = extract_from_page(Page.from_text(OCR_INVOICE))
    assert f.fields["vin"].source == "text"
    f = extract_from_page(ocr_page(OCR_INVOICE), vpic=False)
    assert f.fields["make"].source == "ocr:test"


# --- PDF: OCR fills fields missing from the text layer ------------------------------


def test_pdf_missing_field_is_filled_by_ocr(tmp_path):
    # Mileage was handwritten, so the text layer only has the label.
    lines = [l for l in BODY_SHOP.splitlines() if not l.startswith("Odometer")] + ["Odometer:"]
    path = _pdf(tmp_path, lines)
    ocr = FakeOcr([Seg("Odometer:", 10, 500, 120, 520, score=0.9), Seg("84,211", 130, 500, 200, 520, score=0.9)])
    f = inv.extract_invoice(path, ocr=ocr, vpic=False)
    assert ocr.calls == 1
    assert f.value("mileage") == 84211
    assert f.fields["mileage"].source == "ocr:fake"
    assert f.fields["vin"].source == "pdf_text"  # text-layer fields are kept
    assert f.vin_trusted


def test_pdf_vin_filled_by_ocr_needs_confirmation(tmp_path):
    lines = [l for l in BODY_SHOP.splitlines() if not l.startswith("VIN")]
    path = _pdf(tmp_path, lines)
    ocr = FakeOcr([Seg(f"VIN: {HONDA}", 10, 500, 300, 520, score=0.95)])
    f = inv.extract_invoice(path, ocr=ocr, vpic=False)
    assert f.value("vin") == HONDA
    assert f.fields["vin"].source == "ocr:fake"
    assert not f.vin_trusted
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    f = inv.extract_invoice(path, ocr=ocr, vpic=vpic)
    assert f.vin_trusted


def test_pdf_ocr_fill_does_not_overwrite_text_layer(tmp_path):
    lines = [l for l in BODY_SHOP.splitlines() if not l.startswith("Odometer")]
    path = _pdf(tmp_path, lines)
    ocr = FakeOcr([Seg("Total Due $999.99", 10, 500, 300, 520), Seg("Odometer: 84,211", 10, 540, 300, 560)])
    f = inv.extract_invoice(path, ocr=ocr, vpic=False)
    assert f.value("total") == Decimal("514.50")
    assert f.value("mileage") == 84211


def test_pdf_plate_and_state_filled_together_from_ocr(tmp_path):
    lines = [f"VIN: {HONDA}" if l.startswith("VIN") else l for l in BODY_SHOP.splitlines()]
    path = _pdf(tmp_path, lines)
    ocr = FakeOcr([Seg("License: 7KJD221 CA", 10, 500, 300, 520)])
    f = inv.extract_invoice(path, ocr=ocr, vpic=False)
    assert (f.value("plate"), f.value("plate_state")) == ("7KJD221", "CA")
