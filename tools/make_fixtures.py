"""Generate synthetic vehicle repair invoices with a known answer key.

Each invoice is written three ways:
  <id>.pdf         clean digital PDF with a text layer (what pdfplumber reads)
  <id>_scan.jpg    grayscale, slightly rotated, blurred, noisy, low-quality JPEG
  <id>_photo.jpg   phone photo: perspective skew, uneven lighting, on a tabletop

Difficulty:
  easy (default)  the above; written to tests/fixtures/synthetic/
  hard            same invoices, written to tests/fixtures/synthetic_hard/, plus:
                  mileage handwritten in ink (pen strokes, so not in the PDF text layer),
                  heavy blur, low light, fold creases, and for photos glare and curled paper

Vehicle photos (dev invoices only, in ``dev/photos/``): for each vehicle a door-jamb
sticker with a Code 39 VIN barcode, a dashboard VIN plate, a license-plate close-up and
an odometer display, most with EXIF (capture time, shop GPS, phone model). Labels go to
``photos_answer_key.csv``, plus a few invoice photos labeled shot type "other".
Exterior photos come from tools/fetch_exteriors.py instead.

Matching scenario (dev only, ``dev/matching/`` + ``matching_key.csv``): one photo set per
dev vehicle (exterior from Commons if fetched, plate, VIN plate and/or sticker, odometer),
taken around the invoice date by shop staff, with hard cases: a look-alike car without an
invoice on the same day, sets without VIN or plate shots, an EXIF-less batch upload, a car
nobody invoiced and an invoice without photos. ``--matching-only`` rewrites just these.

Layouts 1-3 go to ``dev/`` for building and tuning extraction. Layouts 4-5 go to
``holdout/`` and are for the final accuracy check only: don't look at them or tune
rules on them, or the final number stops meaning anything.

Output is fully determined by --seed and --difficulty: same files, byte for byte.
Invoice data uses one RNG stream and degradations use per-invoice streams, so easy
and hard contain exactly the same invoices.

VINs come only from tools/vin_pool.json: synthetic VINs (random serials) that NHTSA
vPIC decodes with error code 0 to the intended make, model and year, so a review is
never caused by a VIN no real car could have. The pool is built once, online, by
tools/build_vin_pool.py and committed; generation itself is offline.

Usage:
    python tools/make_fixtures.py [--difficulty easy|hard] [--out DIR] [--count 60] [--seed 7]
                                  [--no-images] [--no-photos] [--matching-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pypdfium2 as pdfium
from fpdf import FPDF
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = {
    "easy": ROOT / "tests" / "fixtures" / "synthetic",
    "hard": ROOT / "tests" / "fixtures" / "synthetic_hard",
}

VIN_POOL_PATH = Path(__file__).resolve().parent / "vin_pool.json"

DEV_LAYOUTS = (1, 2, 3)
HOLDOUT_LAYOUTS = (4, 5)

# --- VINs -----------------------------------------------------------------------
# Deliberately independent of app/extract/vin.py so a bug there can't leak into
# the answer key.

_TRANSLIT = {
    **{str(d): d for d in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
YEAR_CODES = {
    2008: "8", 2009: "9", 2010: "A", 2011: "B", 2012: "C", 2013: "D", 2014: "E",
    2015: "F", 2016: "G", 2017: "H", 2018: "J", 2019: "K", 2020: "L", 2021: "M",
    2022: "N", 2023: "P", 2024: "R", 2025: "S", 2026: "T",
}


def with_check_digit(vin: str) -> str:
    total = sum(_TRANSLIT[c] * w for c, w in zip(vin, _WEIGHTS))
    r = total % 11
    return vin[:8] + ("X" if r == 10 else str(r)) + vin[9:]


@dataclass(frozen=True)
class Model:
    make: str
    model: str
    wmi: str
    vds: str  # positions 4-8: starting pattern for build_vin_pool.py, which corrects it
    plant: str
    years: tuple[int, int]
    trims: tuple[str, ...]


CATALOG = (
    Model("Honda", "Accord", "1HG", "CV1F3", "A", (2018, 2024), ("LX", "Sport", "EX-L")),
    Model("Honda", "Civic", "2HG", "FC2F5", "H", (2016, 2021), ("LX", "EX", "Sport")),
    Model("Honda", "CR-V", "7FA", "RW2H5", "E", (2017, 2022), ("LX", "EX", "Touring")),
    Model("Toyota", "Camry", "4T1", "B11HK", "U", (2018, 2024), ("LE", "SE", "XSE")),
    Model("Toyota", "Corolla", "2T1", "BURHE", "C", (2014, 2019), ("L", "LE", "S")),
    Model("Toyota", "RAV4", "2T3", "W1RFV", "W", (2019, 2024), ("LE", "XLE", "Adventure")),
    Model("Ford", "F-150", "1FT", "EW1EP", "F", (2015, 2023), ("XL", "XLT", "Lariat")),
    Model("Ford", "Escape", "1FM", "CU0GD", "U", (2017, 2022), ("S", "SE", "Titanium")),
    Model("Chevrolet", "Silverado 1500", "3GC", "UYDED", "G", (2016, 2023), ("WT", "LT", "RST")),
    Model("Chevrolet", "Equinox", "2GN", "AXSEV", "L", (2018, 2024), ("LS", "LT", "Premier")),
    Model("Nissan", "Altima", "1N4", "BL4BV", "N", (2016, 2023), ("S", "SV", "SR")),
    Model("Jeep", "Grand Cherokee", "1C4", "RJFAG", "C", (2014, 2021), ("Laredo", "Limited")),
    Model("Subaru", "Outback", "4S4", "BSANC", "3", (2015, 2019), ("2.5i", "Premium", "Limited")),
    Model("Hyundai", "Elantra", "5NP", "D84LF", "H", (2017, 2020), ("SE", "SEL", "Value Edition")),
    Model("Tesla", "Model 3", "5YJ", "3E1EA", "F", (2018, 2024), ("Standard Range", "Long Range")),
    Model("Ram", "1500", "1C6", "SRFFT", "N", (2019, 2024), ("Tradesman", "Big Horn", "Laramie")),
    Model("Volkswagen", "Jetta", "3VW", "C57BU", "M", (2019, 2024), ("S", "SE", "SEL")),
    Model("BMW", "330i", "WBA", "5R1C5", "F", (2019, 2023), ("Base", "xDrive")),
    Model("Mazda", "CX-5", "JM3", "KFBCM", "0", (2017, 2024), ("Sport", "Touring", "Grand Touring")),
)

for _m in CATALOG:
    assert not set(_m.wmi + _m.vds + _m.plant) & set("IOQ"), _m


def load_vin_pool(path: Path = VIN_POOL_PATH) -> dict[tuple[str, str], dict[int, list[str]]]:
    """(make, model) -> {year: [vPIC-clean VINs]}, only years that have VINs."""
    pool = json.loads(path.read_text())["pool"]
    out = {}
    for key, years in pool.items():
        make, model = key.split("|")
        out[(make, model)] = {int(y): vins for y, vins in years.items() if vins}
    return out


# --- Plates, people, shops ------------------------------------------------------

_L = "ABCDEFGHJKLMNPRSTUVWXYZ"  # many states skip I/O/Q on plates too
_D = "0123456789"

PLATE_FORMATS = {  # state -> pattern (L letter, D digit, other chars literal)
    "CA": "DLLLDDD", "TX": "LLLDDDD", "NY": "LLLDDDD", "FL": "LLLLDD",
    "IL": "LLDDDDD", "WA": "LLLDDDD", "AZ": "LLLDDDD", "OH": "LLLDDDD",
}

COLORS = {  # canonical -> printed variants
    "white": ("White", "WHITE", "Pearl White", "WHT"),
    "black": ("Black", "BLACK", "BLK", "Crystal Black Pearl"),
    "silver": ("Silver", "SILVER MET", "Lunar Silver Metallic", "SIL"),
    "gray": ("Gray", "GRAY", "Magnetic Gray", "GRY"),
    "blue": ("Blue", "BLUE", "Deep Blue Metallic", "BLU"),
    "red": ("Red", "RED", "Ruby Red", "RED MET"),
}

FIRST = ("Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn")
LAST = ("Rivera", "Chen", "Patel", "Okafor", "Nguyen", "Kowalski", "Haddad", "Larsen", "Mendez", "Brooks")
STREETS = ("Oak St", "Maple Ave", "Industrial Pkwy", "Commerce Dr", "Mill Rd", "Harbor Blvd")
CITIES = (("Springfield", "IL"), ("Riverside", "CA"), ("Fairview", "TX"), ("Lakewood", "OH"),
          ("Greenville", "WA"), ("Franklin", "AZ"))

# Fictional shop names; 555-01xx is the reserved fictional phone range.
VENDORS = {
    1: "Riverside Collision Center",
    2: "Lakeshore Motors Service Dept.",
    3: "QuickStop Lube & Tire",
    4: "Summit Auto Body & Paint",
    5: "Northgate Fleet Services",
}

# (description, part number or None, unit price range, labor hours range)
REPAIRS = (
    ("Front bumper cover", "04711-TVA-A00ZZ", (180, 520), (1.5, 3.0)),
    ("Headlamp assembly, left", "33150-TVA-A01", (240, 690), (0.8, 1.2)),
    ("Grille assembly", "53111-06830", (120, 340), (0.5, 1.0)),
    ("Rear bumper reinforcement", "52171-06200", (90, 260), (1.0, 2.0)),
    ("Fender, right front", "60211-TVA-A90ZZ", (150, 380), (1.8, 2.6)),
    ("Door shell, left rear", "67550-TBA-A00ZZ", (410, 900), (3.0, 4.5)),
    ("Side mirror, power, right", "76208-TVA-A21", (160, 420), (0.4, 0.8)),
    ("Windshield, acoustic", "73111-TVA-A01", (330, 780), (1.5, 2.0)),
    ("Brake pads, front", "45022-TVA-A01", (55, 140), (1.0, 1.5)),
    ("Brake rotors, front (pair)", "45251-TVA-A00", (110, 260), (0.5, 0.8)),
    ("Battery, group 51R", None, (140, 230), (0.3, 0.5)),
    ("Engine air filter", "17220-6B2-A00", (18, 45), (0.2, 0.3)),
    ("Cabin air filter", "80292-TBA-A11", (20, 55), (0.2, 0.3)),
    ("Synthetic oil 0W-20, 5 qt", None, (32, 60), (0.5, 0.5)),
    ("Oil filter", "15400-PLM-A02", (8, 18), (0.0, 0.0)),
    ("Wiper blades (pair)", None, (25, 55), (0.2, 0.2)),
    ("Tire, 235/45R18 all-season", None, (150, 260), (0.4, 0.5)),
    ("Wheel alignment, 4-wheel", None, (0, 0), (1.0, 1.2)),
    ("Refinish bumper, blend fender", None, (0, 0), (2.5, 4.0)),
    ("Paint materials", None, (85, 240), (0.0, 0.0)),
)

CENT = Decimal("0.01")


def money(x: Decimal) -> str:
    return f"{x.quantize(CENT, ROUND_HALF_UP):,.2f}"


@dataclass
class LineItem:
    kind: str  # part | labor | material
    description: str
    part_no: str | None
    qty: Decimal
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return (self.qty * self.unit_price).quantize(CENT, ROUND_HALF_UP)


@dataclass
class Invoice:
    invoice_id: str
    layout: int
    vendor: str
    vendor_address: str
    vendor_phone: str
    invoice_no: str
    invoice_date: date
    customer: str
    model: Model
    year: int
    trim: str
    vin: str | None
    plate: str | None
    plate_state: str | None
    color: str | None  # canonical
    color_printed: str | None
    mileage: int
    items: list[LineItem] = field(default_factory=list)
    labor_rate: Decimal = Decimal("0")
    tax_rate: Decimal = Decimal("0")
    claim_no: str | None = None

    @property
    def parts_total(self) -> Decimal:
        return sum((i.amount for i in self.items if i.kind != "labor"), Decimal("0"))

    @property
    def labor_total(self) -> Decimal:
        return sum((i.amount for i in self.items if i.kind == "labor"), Decimal("0"))

    @property
    def tax(self) -> Decimal:  # parts and materials taxed, labor not
        return (self.parts_total * self.tax_rate).quantize(CENT, ROUND_HALF_UP)

    @property
    def total(self) -> Decimal:
        return self.parts_total + self.labor_total + self.tax


def make_plate(rng: random.Random, state: str) -> str:
    return "".join(rng.choice(_L) if c == "L" else rng.choice(_D) if c == "D" else c
                   for c in PLATE_FORMATS[state])


def make_invoice(rng: random.Random, n: int, layout: int,
                 pool: dict[tuple[str, str], dict[int, list[str]]], used: set[str]) -> Invoice:
    model = rng.choice([m for m in CATALOG if pool.get((m.make, m.model))])
    years = pool[(model.make, model.model)]
    year = rng.choice(sorted(years))
    fresh = [v for v in years[year] if v not in used] or years[year]
    vin = rng.choice(fresh)
    used.add(vin)
    city, state = rng.choice(CITIES)
    plate_state = state if rng.random() < 0.85 else rng.choice(list(PLATE_FORMATS))
    has_plate = rng.random() < 0.85
    # The quick-lube layout sometimes skips the VIN and identifies by plate only.
    has_vin = not (layout == 3 and rng.random() < 0.2)
    if not has_vin:
        has_plate = True
    has_color = rng.random() < 0.8
    color = rng.choice(list(COLORS)) if has_color else None

    age = max(2026 - year, 0)
    mileage = rng.randint(max(age * 6000, 1500), max(age * 16000, 9000))
    invoice_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 637))

    inv = Invoice(
        invoice_id=f"inv_{n:03d}",
        layout=layout,
        vendor=VENDORS[layout],
        vendor_address=f"{rng.randint(100, 9899)} {rng.choice(STREETS)}, {city}, {state} {rng.randint(10000, 99999)}",
        vendor_phone=f"({rng.randint(201, 989)}) 555-01{rng.randint(0, 99):02d}",
        invoice_no=_invoice_no(rng, layout),
        invoice_date=invoice_date,
        customer=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        model=model,
        year=year,
        trim=rng.choice(model.trims),
        vin=vin if has_vin else None,
        plate=make_plate(rng, plate_state) if has_plate else None,
        plate_state=plate_state if has_plate else None,
        color=color,
        color_printed=rng.choice(COLORS[color]) if color else None,
        mileage=mileage,
        labor_rate=Decimal(rng.choice((95, 115, 125, 139, 149, 165))),
        tax_rate=Decimal(rng.choice(("0.0625", "0.0725", "0.0825", "0.0875"))),
        claim_no=f"CLM-{invoice_date.year}-{rng.randint(1000000, 9999999)}" if layout in (1, 4) and rng.random() < 0.6 else None,
    )
    pool = REPAIRS[8:16] if layout == 3 else REPAIRS
    for desc, part_no, price, hours in rng.sample(pool, rng.randint(2, 5)):
        if price[1]:
            kind = "material" if desc == "Paint materials" else "part"
            qty = Decimal(rng.choice((1, 1, 1, 2))) if kind == "part" else Decimal(1)
            unit = Decimal(rng.randint(price[0] * 100, price[1] * 100)) / 100
            inv.items.append(LineItem(kind, desc, part_no, qty, unit))
        if hours[1]:
            h = Decimal(str(round(rng.uniform(*hours), 1)))
            inv.items.append(LineItem("labor", f"Labor: {desc[0].lower()}{desc[1:]}", None, h, inv.labor_rate))
    return inv


def _invoice_no(rng: random.Random, layout: int) -> str:
    return {
        1: lambda: f"RC-{rng.randint(10000, 99999)}",
        2: lambda: f"RO {rng.randint(400000, 499999)}",
        3: lambda: f"{rng.randint(1000000, 9999999)}",
        4: lambda: f"SAB{rng.randint(2025, 2026)}-{rng.randint(1000, 9999)}",
        5: lambda: f"NFS-{rng.randint(100000, 999999)}",
    }[layout]()


# --- PDF layouts ----------------------------------------------------------------


# Stroke shapes for handwritten digits in a unit box (x right, y down). Each glyph is
# a list of polylines.
def _ellipse(cx, cy, rx, ry, n=14):
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


GLYPHS = {
    "0": [_ellipse(0.5, 0.5, 0.38, 0.5)],
    "1": [[(0.3, 0.2), (0.55, 0.0), (0.55, 1.0)]],
    "2": [[(0.1, 0.25), (0.3, 0.02), (0.7, 0.02), (0.9, 0.25), (0.85, 0.45), (0.1, 1.0), (0.95, 1.0)]],
    "3": [[(0.1, 0.1), (0.4, 0.0), (0.85, 0.15), (0.8, 0.4), (0.4, 0.5), (0.85, 0.6), (0.9, 0.85), (0.5, 1.0), (0.1, 0.9)]],
    "4": [[(0.7, 1.0), (0.7, 0.0), (0.05, 0.7), (0.95, 0.7)]],
    "5": [[(0.9, 0.0), (0.2, 0.0), (0.15, 0.45), (0.6, 0.4), (0.9, 0.6), (0.85, 0.9), (0.5, 1.0), (0.1, 0.9)]],
    "6": [[(0.8, 0.05), (0.4, 0.1), (0.15, 0.5), (0.15, 0.85), (0.45, 1.0), (0.8, 0.9), (0.85, 0.65), (0.5, 0.5), (0.15, 0.65)]],
    "7": [[(0.05, 0.0), (0.95, 0.0), (0.4, 1.0)]],
    "8": [_ellipse(0.5, 0.25, 0.3, 0.25), _ellipse(0.5, 0.72, 0.38, 0.28)],
    "9": [[(0.85, 0.35), (0.5, 0.5), (0.15, 0.35), (0.2, 0.08), (0.55, 0.0), (0.85, 0.15), (0.85, 0.35), (0.75, 1.0)]],
    ",": [[(0.5, 0.85), (0.3, 1.2)]],
}


def handwrite(pdf: FPDF, x: float, y: float, text: str, size: float, rng: random.Random) -> None:
    """Draw digits as jittery, slanted pen strokes in blue-black ink. Vector lines,
    not text, so a PDF text layer doesn't contain them."""
    h = size * 0.3528 * rng.uniform(0.95, 1.25)  # pt -> mm, a bit bigger than print
    slant = rng.uniform(0.1, 0.3)
    pdf.set_draw_color(25, 30, rng.randint(70, 120))
    pdf.set_line_width(rng.uniform(0.3, 0.5))
    cx = x
    for ch in text:
        if ch not in GLYPHS:
            cx += 0.4 * h
            continue
        w = (0.35 if ch == "," else 0.62) * h
        base = y + rng.uniform(-0.12, 0.12) * h
        scale = rng.uniform(0.9, 1.1)
        for stroke in GLYPHS[ch]:
            pts = []
            for px, py in stroke:
                jx, jy = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
                yy = base + (py + jy) * h * scale
                pts.append((cx + (px + jx) * w + (1 - py) * slant * h, yy))
            pdf.polyline(pts)
        cx += w + rng.uniform(0.12, 0.28) * h
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.2)


class Doc(FPDF):
    hand: random.Random | None = None  # set in hard mode: mileage is handwritten

    def text_at(self, x, y, s, size=10, style="", font="Helvetica", w=0, align="L", h=5):
        self.set_font(font, style, size)
        self.set_xy(x, y)
        self.cell(w, h, s, align=align)

    def value_at(self, x, y, s, size=10, style="", font="Helvetica"):
        """A value someone might fill in by hand (mileage)."""
        if self.hand is None:
            self.text_at(x, y, s, size=size, style=style, font=font)
        else:
            handwrite(self, x + 0.5, y + 0.4, s, size, self.hand)

    def label_value(self, x, y, label, value, size=10, style="", font="Helvetica"):
        self.text_at(x, y, label, size=size, style=style, font=font)
        self.set_font(font, style, size)
        self.value_at(x + self.get_string_width(label), y, value, size=size, style=style, font=font)


def new_doc(hand: random.Random | None = None) -> Doc:
    pdf = Doc(unit="mm", format="Letter")
    pdf.hand = hand
    pdf.set_auto_page_break(False)
    pdf.set_creation_date(datetime(2026, 1, 1))
    pdf.add_page()
    return pdf


def _items_table(pdf: Doc, inv: Invoice, y: float, cols, font="Helvetica", size=9, header_fill=None, zebra=False):
    """cols: list of (title, width, getter, align)."""
    x0 = 15
    pdf.set_font(font, "B", size)
    if header_fill:
        pdf.set_fill_color(*header_fill)
    x = x0
    for title, w, _, align in cols:
        pdf.set_xy(x, y)
        pdf.cell(w, 7, title, border="B", align=align, fill=bool(header_fill))
        x += w
    y += 8
    pdf.set_font(font, "", size)
    for n, item in enumerate(inv.items):
        if zebra and n % 2:
            pdf.set_fill_color(242, 242, 242)
            pdf.rect(x0, y - 0.5, sum(c[1] for c in cols), 6, style="F")
        x = x0
        for _, w, get, align in cols:
            pdf.set_xy(x, y)
            pdf.cell(w, 5, get(item), align=align)
            x += w
        y += 6
    return y


def _totals(pdf: Doc, inv: Invoice, y: float, x=130, font="Helvetica", labels=("Parts", "Labor", "Tax", "TOTAL")):
    rows = [(labels[0], inv.parts_total), (labels[1], inv.labor_total),
            (f"{labels[2]} ({inv.tax_rate * 100:.2f}%)", inv.tax), (labels[3], inv.total)]
    for n, (label, value) in enumerate(rows):
        last = n == len(rows) - 1
        pdf.text_at(x, y, label, size=10 if not last else 11, style="B" if last else "", font=font)
        pdf.text_at(x + 30, y, f"${money(value)}", size=10 if not last else 11, style="B" if last else "",
                    font=font, w=35, align="R")
        y += 6
    return y


def layout_1(inv: Invoice, hand: random.Random | None = None) -> Doc:
    """Body shop invoice: shop header left, invoice box right, vehicle grid, item table."""
    pdf = new_doc(hand)
    pdf.text_at(15, 15, inv.vendor, size=18, style="B")
    pdf.text_at(15, 23, inv.vendor_address, size=9)
    pdf.text_at(15, 28, f"Phone {inv.vendor_phone}", size=9)
    pdf.rect(135, 14, 65, 22)
    pdf.text_at(138, 16, "INVOICE", size=14, style="B")
    pdf.text_at(138, 24, f"Invoice #: {inv.invoice_no}", size=9)
    pdf.text_at(138, 29, f"Date: {inv.invoice_date:%m/%d/%Y}", size=9)

    pdf.text_at(15, 42, f"Bill to: {inv.customer}", size=10)
    if inv.claim_no:
        pdf.text_at(110, 42, f"Insurance claim: {inv.claim_no}", size=9)

    pdf.set_fill_color(230, 236, 245)
    pdf.rect(15, 50, 185, 26, style="F")
    pdf.text_at(18, 51, "VEHICLE INFORMATION", size=9, style="B")
    grid = [  # (x, y, label, value)
        (18, 58, "Year", str(inv.year)), (64, 58, "Make", inv.model.make),
        (110, 58, "Model", inv.model.model), (156, 58, "Trim", inv.trim),
        (18, 67, "VIN", inv.vin or ""), (82, 67, "Mileage", f"{inv.mileage:,}"),
        (122, 67, "Plate", f"{inv.plate} ({inv.plate_state})" if inv.plate else ""),
        (163, 67, "Color", inv.color_printed or ""),
    ]
    for x, y, label, value in grid:
        if label == "Mileage":
            pdf.label_value(x, y, "Mileage: ", value, size=9)
        else:
            pdf.text_at(x, y, f"{label}: {value}", size=9)

    y = _items_table(pdf, inv, 84, [
        ("Description", 85, lambda i: i.description, "L"),
        ("Part #", 40, lambda i: i.part_no or "", "L"),
        ("Qty/Hrs", 18, lambda i: f"{i.qty}", "R"),
        ("Unit", 20, lambda i: money(i.unit_price), "R"),
        ("Amount", 22, lambda i: money(i.amount), "R"),
    ])
    _totals(pdf, inv, y + 8)
    pdf.text_at(15, 250, "Thank you for your business. All work guaranteed 12 months / 12,000 miles.", size=8)
    return pdf


def layout_2(inv: Invoice, hand: random.Random | None = None) -> Doc:
    """Dealer service repair order: dense boxed fields, V.I.N. label, mileage in/out."""
    pdf = new_doc(hand)
    pdf.set_draw_color(60, 60, 60)
    pdf.text_at(15, 12, inv.vendor.upper(), size=14, style="B")
    pdf.text_at(15, 19, f"{inv.vendor_address}  |  {inv.vendor_phone}", size=8)
    pdf.text_at(150, 12, "REPAIR ORDER", size=12, style="B", w=50, align="R")
    pdf.text_at(150, 19, inv.invoice_no, size=12, w=50, align="R")

    def box(x, y, w, label, value, filled_in=False):
        pdf.rect(x, y, w, 12)
        pdf.text_at(x + 1, y + 0.5, label, size=6, style="B")
        (pdf.value_at if filled_in else pdf.text_at)(x + 1, y + 5.5, value, size=10)

    mi_out = inv.mileage + random.Random(inv.invoice_id).randint(0, 12)
    box(15, 28, 45, "R.O. DATE", f"{inv.invoice_date:%Y-%m-%d}")
    box(60, 28, 70, "CUSTOMER", inv.customer.upper())
    box(130, 28, 70, "ADVISOR", random.Random(inv.invoice_no).choice(LAST).upper())
    box(15, 40, 70, "V.I.N.", inv.vin or "")
    box(85, 40, 60, "YR/MAKE/MODEL", f"{inv.year} {inv.model.make} {inv.model.model} {inv.trim}".upper())
    box(145, 40, 55, "COLOR", (inv.color_printed or "").upper())
    box(15, 52, 45, "LIC#", f"{inv.plate_state} {inv.plate}" if inv.plate else "")
    box(60, 52, 35, "MILEAGE IN", f"{inv.mileage}", filled_in=True)
    box(95, 52, 35, "MILEAGE OUT", f"{mi_out}", filled_in=True)
    box(130, 52, 70, "TAG / HAT", f"T{random.Random(inv.vendor_phone).randint(100, 999)}")

    y = _items_table(pdf, inv, 72, [
        ("OP", 15, lambda i: {"labor": "L", "part": "P", "material": "M"}[i.kind], "L"),
        ("DESCRIPTION", 95, lambda i: i.description.upper(), "L"),
        ("PART NO.", 35, lambda i: i.part_no or "", "L"),
        ("QTY", 15, lambda i: f"{i.qty}", "R"),
        ("AMOUNT", 25, lambda i: money(i.amount), "R"),
    ], size=8, header_fill=(220, 220, 220))
    _totals(pdf, inv, y + 8, labels=("PARTS", "LABOR", "SALES TAX", "TOTAL DUE"))
    pdf.text_at(15, 240, "CUSTOMER SIGNATURE ________________________________", size=9)
    return pdf


def layout_3(inv: Invoice, hand: random.Random | None = None) -> Doc:
    """Quick-lube receipt: narrow, monospaced, VIN label on the line above the VIN."""
    pdf = new_doc(hand)
    x, w = 60, 95
    f = "Courier"
    y = 15
    for line, size, style in [
        (inv.vendor.upper(), 12, "B"), (inv.vendor_address, 8, ""), (inv.vendor_phone, 8, ""),
    ]:
        pdf.text_at(x, y, line, size=size, style=style, font=f, w=w, align="C")
        y += 6
    y += 3
    rows = [f"RECEIPT  {inv.invoice_no}", f"DATE     {inv.invoice_date:%m/%d/%y}", f"CUST     {inv.customer}", "",
            f"{inv.year} {inv.model.make} {inv.model.model}"]
    if inv.plate:
        rows.append(f"PLATE    {inv.plate} {inv.plate_state}")
    rows.append(f"ODO      {inv.mileage}")
    if inv.vin:
        rows += ["VEHICLE ID", inv.vin]
    if inv.color_printed:
        rows.append(f"COLOR    {inv.color_printed}")
    rows.append("-" * 38)
    for r in rows:
        if r.startswith("ODO "):
            pdf.label_value(x, y, "ODO      ", r.split()[-1], size=9, font=f)
        else:
            pdf.text_at(x, y, r, size=9, font=f)
        y += 5
    for item in inv.items:
        pdf.text_at(x, y, item.description[:26], size=9, font=f)
        pdf.text_at(x, y, money(item.amount), size=9, font=f, w=w, align="R")
        y += 5
        if item.qty != 1:
            pdf.text_at(x + 4, y, f"{item.qty} @ {money(item.unit_price)}", size=8, font=f)
            y += 5
    pdf.text_at(x, y, "-" * 38, size=9, font=f)
    y += 5
    for label, value in [("SUBTOTAL", inv.parts_total + inv.labor_total), ("TAX", inv.tax), ("TOTAL", inv.total)]:
        pdf.text_at(x, y, label, size=9 if label != "TOTAL" else 11, style="B" if label == "TOTAL" else "", font=f)
        pdf.text_at(x, y, money(value), size=9 if label != "TOTAL" else 11, style="B" if label == "TOTAL" else "",
                    font=f, w=w, align="R")
        y += 6
    pdf.text_at(x, y + 4, "NEXT SERVICE DUE AT " + f"{inv.mileage + 5000:,}" + " MI", size=8, font=f, w=w, align="C")
    return pdf


def layout_4(inv: Invoice, hand: random.Random | None = None) -> Doc:
    """HOLDOUT. Insurance-style estimate/invoice: serif font, VIN printed in three
    space-separated groups, vehicle block at the bottom, long-form date."""
    pdf = new_doc(hand)
    f = "Times"
    pdf.text_at(15, 15, inv.vendor, size=20, style="B", font=f)
    pdf.text_at(15, 24, inv.vendor_address + " - Tel " + inv.vendor_phone, size=9, font=f)
    pdf.line(15, 31, 200, 31)
    pdf.text_at(15, 34, "Final Invoice", size=13, style="B", font=f)
    pdf.text_at(120, 34, f"No. {inv.invoice_no}", size=11, font=f, w=80, align="R")
    pdf.text_at(120, 40, f"{inv.invoice_date:%d %B %Y}", size=11, font=f, w=80, align="R")
    pdf.text_at(15, 42, f"Owner: {inv.customer}", size=10, font=f)
    if inv.claim_no:
        pdf.text_at(15, 48, f"Claim number {inv.claim_no}", size=10, font=f)

    y = _items_table(pdf, inv, 58, [
        ("Operation", 110, lambda i: i.description, "L"),
        ("Hrs/Qty", 25, lambda i: f"{i.qty}", "R"),
        ("Price", 25, lambda i: money(i.unit_price), "R"),
        ("Total", 25, lambda i: money(i.amount), "R"),
    ], font=f, size=10, zebra=True)
    y = _totals(pdf, inv, y + 6, font=f, labels=("Parts & materials", "Labor", "Tax", "Amount due"))

    y = max(y + 12, 190)
    pdf.line(15, y, 200, y)
    pdf.text_at(15, y + 2, "Vehicle", size=11, style="B", font=f)
    vin_groups = f"{inv.vin[:5]} {inv.vin[5:10]} {inv.vin[10:]}" if inv.vin else ""
    lines = [
        f"{inv.model.make} {inv.model.model} {inv.trim} ({inv.year})",
        f"Identification: {vin_groups}",
        f"Registration: {inv.plate or '-'}" + (f", {inv.plate_state}" if inv.plate else ""),
        f"Odometer reading: {inv.mileage:,} miles",
    ]
    if inv.color_printed:
        lines.append(f"Paint: {inv.color_printed}")
    for n, line in enumerate(lines):
        if line.startswith("Odometer reading: "):
            pdf.label_value(15, y + 9 + n * 6, "Odometer reading: ", line.split(": ")[1].removesuffix(" miles"),
                            size=10, font=f)
        else:
            pdf.text_at(15, y + 9 + n * 6, line, size=10, font=f)
    return pdf


def layout_5(inv: Invoice, hand: random.Random | None = None) -> Doc:
    """HOLDOUT. Fleet services statement: dark header band, vehicle as a table row
    with column headers above (Serial #, Unit Yr, ...), ISO-ish date, odometer in km-
    style 'mi' column."""
    pdf = new_doc(hand)
    pdf.set_fill_color(35, 45, 60)
    pdf.rect(0, 0, 216, 30, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.text_at(15, 8, inv.vendor, size=16, style="B")
    pdf.text_at(15, 17, inv.vendor_address, size=8)
    pdf.text_at(140, 8, "SERVICE STATEMENT", size=11, style="B", w=60, align="R")
    pdf.text_at(140, 15, f"Ref {inv.invoice_no}", size=9, w=60, align="R")
    pdf.text_at(140, 20, f"Issued {inv.invoice_date:%b %d, %Y}", size=9, w=60, align="R")
    pdf.set_text_color(0, 0, 0)
    pdf.text_at(15, 36, f"Account: {inv.customer} / Fleet", size=10)

    cols = [("Serial #", 48), ("Unit Yr", 16), ("Make", 24), ("Model", 34), ("Plate/St", 28), ("Odometer", 20), ("Color", 15)]
    values = [inv.vin or "", str(inv.year), inv.model.make, inv.model.model,
              f"{inv.plate}/{inv.plate_state}" if inv.plate else "", f"{inv.mileage}", (inv.color_printed or "")[:8]]
    x, y = 15, 46
    pdf.set_fill_color(235, 235, 235)
    for (title, w), value in zip(cols, values):
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_xy(x, y)
        pdf.cell(w, 6, title, border=1, fill=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(x, y + 6)
        if title == "Odometer":
            pdf.cell(w, 7, "", border=1)
            pdf.value_at(x + 0.5, y + 7, value, size=8)
        else:
            pdf.cell(w, 7, value, border=1)
        x += w

    y = _items_table(pdf, inv, 68, [
        ("Type", 20, lambda i: i.kind.title(), "L"),
        ("Description", 95, lambda i: i.description, "L"),
        ("Qty", 20, lambda i: f"{i.qty}", "R"),
        ("Rate", 25, lambda i: money(i.unit_price), "R"),
        ("Ext.", 25, lambda i: money(i.amount), "R"),
    ], size=9, zebra=True)
    _totals(pdf, inv, y + 8, labels=("Parts", "Labor", "Tax", "Balance"))
    pdf.text_at(15, 245, "Net 30. Questions? Contact fleet billing at " + inv.vendor_phone, size=8)
    return pdf


LAYOUTS = {1: layout_1, 2: layout_2, 3: layout_3, 4: layout_4, 5: layout_5}


# --- Image degradation ----------------------------------------------------------


def render(pdf_path: Path, dpi: int) -> Image.Image:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return doc[0].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        doc.close()


def add_noise(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    """Seeded grain: a uniform-noise tile from rng (std scaled to sigma), tiled over
    the image. Unlike Image.effect_noise this is reproducible."""
    tile = Image.frombytes("L", (256, 256), rng.randbytes(256 * 256))
    tile = tile.point(lambda p: max(0, min(255, round(128 + (p - 128) * sigma / 73.9))))
    noise = Image.new("L", img.size)
    ox, oy = rng.randrange(256), rng.randrange(256)
    for x in range(-ox, img.width, 256):
        for y in range(-oy, img.height, 256):
            noise.paste(tile, (x, y))
    if img.mode == "RGB":
        noise = Image.merge("RGB", (noise, noise, noise))
    return ImageChops.add(img, noise, scale=1.0, offset=-128)


def add_folds(img: Image.Image, rng: random.Random, n: int) -> Image.Image:
    """Fold creases: a dark/light line pair across the page, slightly darker on one side."""
    img = img.copy()
    w, h = img.size
    dark = (110,) * (3 if img.mode == "RGB" else 1)
    light = (250,) * (3 if img.mode == "RGB" else 1)
    for _ in range(n):
        horizontal = rng.random() < 0.7
        pos = rng.uniform(0.3, 0.7) * (h if horizontal else w)
        tilt = rng.uniform(-0.02, 0.02) * (w if horizontal else h)
        if horizontal:
            line, side = [(0, pos), (w, pos + tilt)], [(0, pos), (w, pos + tilt), (w, h), (0, h)]
        else:
            line, side = [(pos, 0), (pos + tilt, h)], [(pos, 0), (pos + tilt, h), (w, h), (w, 0)]
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(side, fill=255)
        img = Image.composite(ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 0.93)), img, mask)
        draw = ImageDraw.Draw(img)
        draw.line(line, fill=dark, width=max(2, w // 600))
        offset = [(x + (0 if horizontal else 3), y + (3 if horizontal else 0)) for x, y in line]
        draw.line(offset, fill=light, width=max(2, w // 800))
    return img


def add_glare(img: Image.Image, rng: random.Random) -> Image.Image:
    """A bright elliptical hotspot (flash or window reflection) that washes out text."""
    w, h = img.size
    gw, gh = int(w * rng.uniform(0.25, 0.45)), int(h * rng.uniform(0.12, 0.25))
    spot = ImageOps.invert(Image.radial_gradient("L")).resize((gw, gh), Image.BICUBIC)
    strength = rng.uniform(0.75, 0.95)
    spot = spot.point(lambda p: int(min(255, max(0, (p - 60) * 1.6)) * strength))
    mask = Image.new("L", img.size, 0)
    mask.paste(spot, (int(rng.uniform(0.05, 0.6) * w), int(rng.uniform(0.05, 0.7) * h)))
    return Image.composite(Image.new(img.mode, img.size, (255, 255, 250)), img, mask)


def curl(img: Image.Image, rng: random.Random, fill) -> Image.Image:
    """Curled paper: rows bend near one edge (mesh warp), like a page lifting off a table."""
    w, h = img.size
    amp = rng.uniform(0.03, 0.06) * h
    edge = rng.choice(("left", "right"))

    def src(x: float, y: float) -> tuple[float, float]:
        t = (x / w) if edge == "right" else (1 - x / w)
        lift = t ** 3
        return x - lift * 0.02 * w * (1 if edge == "right" else -1), y + amp * lift * (0.5 - y / h) * 2

    mesh, nx, ny = [], 16, 20
    for i in range(nx):
        for j in range(ny):
            x0, x1 = w * i / nx, w * (i + 1) / nx
            y0, y1 = h * j / ny, h * (j + 1) / ny
            quad = (*src(x0, y0), *src(x0, y1), *src(x1, y1), *src(x1, y0))
            mesh.append(((int(x0), int(y0), int(x1), int(y1)), quad))
    return img.transform(img.size, Image.MESH, mesh, Image.BICUBIC, fillcolor=fill)


def make_scan(pdf_path: Path, out: Path, rng: random.Random, hard: bool = False) -> None:
    img = render(pdf_path, 150).convert("L")
    if hard:
        img = add_folds(img, rng, rng.randint(1, 2))
    img = img.rotate(rng.uniform(-3, 3) if hard else rng.uniform(-1.5, 1.5), resample=Image.BICUBIC,
                     expand=True, fillcolor=238)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.1, 1.5) if hard else rng.uniform(0.7, 1.3)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 0.65) if hard else rng.uniform(0.7, 0.85))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 0.88) if hard else rng.uniform(1.0, 1.08))
    img = add_noise(img, rng.uniform(14, 20) if hard else rng.uniform(8, 14), rng)
    img.save(out, "JPEG", quality=rng.randint(30, 42) if hard else rng.randint(45, 60))


def _perspective_coeffs(dst, src):
    """Coefficients for Image.transform(PERSPECTIVE) mapping output points dst to
    input points src (4 pairs). Plain Gaussian elimination, no numpy."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    a = [r + [b] for r, b in zip(rows, rhs)]
    n = 8
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c:
                k = a[r][c] / a[c][c]
                a[r] = [vr - k * vc for vr, vc in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def make_photo(pdf_path: Path, out: Path, rng: random.Random, hard: bool = False) -> None:
    paper = render(pdf_path, 170)
    if hard:
        paper = add_folds(paper, rng, rng.randint(1, 2))
    # Warm paper tint and uneven lighting (one side darker, angle random).
    paper = Image.blend(paper, Image.new("RGB", paper.size, (250, 240, 222)), rng.uniform(0.08, 0.2))
    # Rotate a gradient, then crop inside the inscribed circle so no black corners remain.
    shade = Image.linear_gradient("L").resize((512, 512)).rotate(rng.uniform(0, 360), Image.BICUBIC)
    shade = shade.crop((256 - 181, 256 - 181, 256 + 181, 256 + 181)).resize(paper.size, Image.BICUBIC)
    shade = ImageOps.autocontrast(shade).point(lambda p: 90 + p * 165 // 255)
    paper = Image.composite(paper, ImageEnhance.Brightness(paper).enhance(rng.uniform(0.55, 0.75)), shade)

    w, h = paper.size
    W, H = int(w * 1.25), int(h * 1.2)
    bg_color = rng.choice(((112, 86, 62), (60, 62, 66), (168, 160, 150), (40, 70, 60)))
    canvas = add_noise(Image.new("RGB", (W, H), bg_color), 10, rng).filter(ImageFilter.GaussianBlur(2))
    paper_mask = Image.new("L", paper.size, 255)
    if hard:
        paper = add_glare(paper, rng)
        state = rng.getstate()
        paper = curl(paper, rng, fill=bg_color)
        rng.setstate(state)  # same warp for the mask
        paper_mask = curl(paper_mask, rng, fill=0)

    def jitter(px, py, s=0.1 if hard else 0.07):
        return (px + rng.uniform(-s, s) * W, py + rng.uniform(-s, s) * H)

    mx, my = (W - w) / 2, (H - h) / 2
    dst = [jitter(mx, my), jitter(mx + w, my), jitter(mx + w, my + h), jitter(mx, my + h)]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs = _perspective_coeffs(dst, src)
    warped = paper.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    mask = paper_mask.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    canvas.paste(warped, (0, 0), mask)

    canvas = canvas.rotate(rng.uniform(-6, 6) if hard else rng.uniform(-4, 4), resample=Image.BICUBIC,
                           fillcolor=bg_color)
    if hard:  # low light: dim, warm indoor tint, more sensor noise
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.45, 0.62))
        canvas = Image.blend(canvas, Image.new("RGB", canvas.size, (255, 170, 80)), rng.uniform(0.06, 0.12))
    canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(1.3, 2.0) if hard else rng.uniform(0.6, 1.2)))
    canvas = add_noise(canvas, rng.uniform(10, 16) if hard else rng.uniform(4, 8), rng)
    canvas.thumbnail((1600, 2000))
    canvas.save(out, "JPEG", quality=rng.randint(55, 70) if hard else rng.randint(72, 85))


# --- Vehicle photos ----------------------------------------------------------------
# For each dev invoice's vehicle: a door-jamb sticker (with a Code 39 VIN barcode), a
# dashboard VIN plate, a license-plate close-up and an odometer display. Exterior photos
# can't be drawn convincingly; tools/fetch_exteriors.py downloads real ones instead.
# Photos use their own per-invoice RNG streams, so the invoices don't change.

PHOTO_KINDS = {"sticker": "door_jamb", "vinplate": "vin_plate", "plate": "plate", "odometer": "odometer"}
PHOTO_KEY_FIELDS = [
    "photo", "invoice_id", "split", "shot_type", "vin", "plate", "plate_state", "odometer", "color",
    "body_style", "captured_at", "uploaded_at", "gps_lat", "gps_lon", "device", "source",
]
OTHER_PHOTOS = 6  # invoice photos of the first dev invoices, labeled shot type "other"
PAINT = {"white": (232, 233, 230), "black": (28, 29, 32), "silver": (176, 179, 183), "gray": (98, 101, 106),
         "blue": (32, 62, 138), "red": (158, 28, 32), None: (120, 122, 126)}
STATE_NAMES = {"CA": "California", "TX": "TEXAS", "NY": "NEW YORK", "FL": "Florida", "IL": "Illinois",
               "WA": "Washington", "AZ": "ARIZONA", "OH": "OHIO"}
STATE_INK = {"CA": (175, 30, 40), "TX": (20, 30, 60), "NY": (20, 50, 140), "FL": (20, 110, 60),
             "IL": (150, 30, 40), "WA": (30, 60, 120), "AZ": (130, 60, 20), "OH": (30, 40, 110)}
BODY_STYLE = {"F-150": "pickup", "Silverado 1500": "pickup", "1500": "pickup", "CR-V": "suv", "RAV4": "suv",
              "Escape": "suv", "Equinox": "suv", "Grand Cherokee": "suv", "CX-5": "suv", "Outback": "wagon"}
DEVICES = (("Apple", "iPhone 13"), ("Apple", "iPhone 15 Pro"), ("samsung", "SM-S911U"), ("Google", "Pixel 7"))


def body_style(model: Model) -> str:
    return BODY_STYLE.get(model.model, "sedan")


def font(size: int):
    return ImageFont.load_default(size=size)  # bundled with Pillow: same glyphs everywhere


def shop_gps(vendor: str) -> tuple[float, float]:
    r = random.Random(f"gps-{vendor}")
    return round(r.uniform(32.0, 47.0), 5), round(r.uniform(-122.0, -80.0), 5)


def _photo_finish(img: Image.Image, rng: random.Random, hard: bool, fill, blur=None) -> Image.Image:
    """Hand-held phone shot: perspective, tilt, blur, noise; hard adds glare and dim light.
    ``blur``: (low, high) Gaussian sigma, default by difficulty."""
    W, H = img.size
    s = 0.07 if hard else 0.04
    dst = [(rng.uniform(-s, s) * W + x, rng.uniform(-s, s) * H + y) for x, y in ((0, 0), (W, 0), (W, H), (0, H))]
    img = img.transform((W, H), Image.PERSPECTIVE, _perspective_coeffs(dst, [(0, 0), (W, 0), (W, H), (0, H)]),
                        Image.BICUBIC, fillcolor=fill)
    img = img.rotate(rng.uniform(-6, 6) if hard else rng.uniform(-3, 3), resample=Image.BICUBIC, fillcolor=fill)
    if hard:
        if rng.random() < 0.5:
            img = add_glare(img, rng)
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.55, 0.8))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*(blur or ((1.2, 2.0) if hard else (0.5, 1.1))))))
    return add_noise(img, rng.uniform(9, 15) if hard else rng.uniform(4, 8), rng)


def _barcode(vin: str, scale: int = 2) -> Image.Image:
    import numpy as np
    import zxingcpp

    code = zxingcpp.create_barcode(vin, zxingcpp.BarcodeFormat.Code39)
    return Image.fromarray(np.asarray(zxingcpp.write_barcode_to_image(code, scale=scale))).convert("RGB")


def draw_sticker(inv: Invoice, vin: str, rng: random.Random, hard: bool) -> tuple[Image.Image, bool]:
    """Door-jamb certification label on painted metal. Returns (image, barcode intact).
    Drawn large enough that the barcode's narrowest bar is ~4 px, as in a phone photo."""
    k = 1.25  # layout scale

    def at(x, y):
        return (round(x * k), round(y * k))

    def f(size):
        return font(round(size * k))

    paint = PAINT[inv.color]
    canvas = Image.new("RGB", (1750, 1300), paint)
    shade = Image.linear_gradient("L").resize(canvas.size).rotate(rng.uniform(0, 360))
    canvas = Image.composite(canvas, ImageEnhance.Brightness(canvas).enhance(0.6), shade)
    label = Image.new("RGB", at(1000, 640), (246, 245, 238))
    d = ImageDraw.Draw(label)
    ink = (25, 25, 30)
    mfd = inv.invoice_date.replace(year=inv.year - 1 if inv.year > 2000 else inv.year)
    d.text(at(30, 22), f"MFD. BY: {inv.model.make.upper()} MOTOR CO.", font=f(30), fill=ink)
    d.text(at(700, 22), f"DATE: {mfd:%m/%y}", font=f(30), fill=ink)
    gvwr = rng.randint(1800, 3200)
    d.text(at(30, 70), f"GVWR: {gvwr} KG ({round(gvwr * 2.2046)} LB)", font=f(24), fill=ink)
    d.text(at(30, 104), f"GAWR FRT: {gvwr // 2 + 40} KG   GAWR RR: {gvwr // 2 - 40} KG", font=f(24), fill=ink)
    for i, line in enumerate(("THIS VEHICLE CONFORMS TO ALL APPLICABLE U.S. FEDERAL MOTOR",
                              "VEHICLE SAFETY, BUMPER, AND THEFT PREVENTION STANDARDS IN",
                              "EFFECT ON THE DATE OF MANUFACTURE SHOWN ABOVE.")):
        d.text(at(30, 150 + 30 * i), line, font=f(21), fill=ink)
    kind = {"pickup": "TRUCK", "suv": "MPV"}.get(body_style(inv.model), "PASSENGER CAR")
    d.text(at(30, 260), f"VIN: {vin}", font=f(40), fill=ink)
    d.text(at(700, 316), f"TYPE: {kind}", font=f(22), fill=ink)
    code = _barcode(vin, scale=4)  # 1064 x 200
    label.paste(code, at(30, 350))
    d.text(at(30, 530 + 60), f"{inv.model.make.upper()} {inv.model.model.upper()}  PAINT {rng.randint(100, 999)}",
           font=f(22), fill=ink)
    intact = True
    if hard and rng.random() < 0.4:  # scuffed or peeling across the barcode
        intact = False
        x = rng.randint(120, 750)
        d.rectangle((*at(x, 340), *at(x + rng.randint(90, 160), 570)), fill=(246, 245, 238))
    label = label.rotate(rng.uniform(-4, 4), expand=True, fillcolor=paint, resample=Image.BICUBIC)
    canvas.paste(label, ((canvas.width - label.width) // 2, (canvas.height - label.height) // 2))
    # A close-up of a label is usually in focus: less blur than the other shots.
    return _photo_finish(canvas, rng, hard, paint, blur=(0.6, 1.3) if hard else (0.4, 0.9)), intact


def draw_vin_plate(inv: Invoice, vin: str, rng: random.Random, hard: bool) -> Image.Image:
    """VIN plate at the base of the windshield, seen through the glass."""
    canvas = Image.new("RGB", (1400, 900), (38, 38, 40))
    d = ImageDraw.Draw(canvas)
    for y in range(0, 900, 6):  # dashboard grain
        g = 34 + (y * 7919 + rng.randint(0, 9)) % 10
        d.line((0, y, 1400, y), fill=(g, g, g + 2))
    plate = Image.new("RGB", (900, 120), (196, 197, 192))
    pd = ImageDraw.Draw(plate)
    pd.rectangle((0, 0, 899, 119), outline=(120, 120, 118), width=4)
    pd.text((40, 26), vin, font=font(64), fill=(30, 30, 32))
    pd.ellipse((10, 50, 26, 66), fill=(90, 90, 90))
    pd.ellipse((872, 50, 888, 66), fill=(90, 90, 90))
    canvas.paste(plate, (250, rng.randint(330, 450)))
    sky = Image.linear_gradient("L").rotate(rng.choice((0, 180)) + rng.uniform(-30, 30)).resize(canvas.size)
    canvas = Image.composite(Image.new("RGB", canvas.size, (210, 220, 235)), canvas,
                             sky.point(lambda p: int(p * (0.35 if hard else 0.18))))
    return _photo_finish(canvas, rng, hard, (38, 38, 40))


def draw_plate(inv: Invoice, plate: str, state: str, rng: random.Random, hard: bool) -> Image.Image:
    """Rear of the car with the license plate filling a good part of the frame."""
    paint = PAINT[inv.color]
    canvas = Image.new("RGB", (1280, 960), paint)
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 560, 1280, 960), fill=tuple(max(0, c - 70) for c in paint))  # bumper
    d.rectangle((0, 880, 1280, 960), fill=(55, 55, 58))  # ground
    shade = Image.linear_gradient("L").resize(canvas.size)
    canvas = Image.composite(canvas, ImageEnhance.Brightness(canvas).enhance(0.7), shade)
    p = Image.new("RGB", (520, 265), (246, 246, 244))
    pd = ImageDraw.Draw(p)
    pd.rounded_rectangle((2, 2, 517, 262), radius=18, outline=(40, 40, 45), width=6)
    ink = STATE_INK.get(state, (30, 30, 60))
    name = STATE_NAMES.get(state, state)
    w = pd.textlength(name, font=font(44))
    pd.text(((520 - w) / 2, 14), name, font=font(44), fill=ink)
    size = 130
    while pd.textlength(plate, font=font(size)) > 470:
        size -= 4
    w = pd.textlength(plate, font=font(size))
    pd.text(((520 - w) / 2, 70 + (130 - size) // 2), plate, font=font(size), fill=(20, 25, 60))
    for x in (60, 440):
        pd.ellipse((x, 20, x + 18, 38), fill=(150, 150, 150))
    canvas.paste(p, (380 + rng.randint(-60, 60), 470 + rng.randint(-30, 30)))
    return _photo_finish(canvas, rng, hard, (55, 55, 58))


def draw_odometer(inv: Invoice, rng: random.Random, hard: bool) -> Image.Image:
    """Instrument cluster: two dials and an LCD with ODO and trip meters."""
    canvas = Image.new("RGB", (1280, 800), (12, 12, 14))
    d = ImageDraw.Draw(canvas)
    lit = (235, 240, 245)
    for cx in (290, 990):
        d.arc((cx - 230, 170, cx + 230, 630), 140, 400, fill=lit, width=5)
        for k in range(13):
            a = math.radians(140 + k * 260 / 12)
            d.line((cx + 200 * math.cos(a), 400 + 200 * math.sin(a), cx + 225 * math.cos(a), 400 + 225 * math.sin(a)),
                   fill=lit, width=4)
        a = math.radians(140 + rng.uniform(0, 30))
        d.line((cx, 400, cx + 170 * math.cos(a), 400 + 170 * math.sin(a)), fill=(240, 60, 40), width=8)
    d.rounded_rectangle((470, 420, 850, 640), radius=14, fill=(18, 42, 46))
    lcd = (190, 245, 235)
    d.text((495, 440), f"TRIP A {rng.uniform(0, 999):.1f}", font=font(30), fill=lcd)
    odo = f"ODO {inv.mileage}"
    d.text((495, 500), odo, font=font(48), fill=lcd)
    d.text((495 + d.textlength(odo, font=font(48)) + 10, 516), "mi", font=font(26), fill=lcd)
    d.text((495, 584), f"{rng.randint(40, 95)}°F   D", font=font(28), fill=lcd)
    return _photo_finish(canvas, rng, hard, (12, 12, 14))


def _exif_bytes(when: datetime, gps: tuple[float, float], device: tuple[str, str]) -> bytes:
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = device
    ex.get_ifd(0x8769)[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")

    def dms(v: float):
        v = abs(v)
        deg, minutes = int(v), int((v % 1) * 60)
        return (deg, minutes, round(((v % 1) * 60 - minutes) * 60, 2))

    g = ex.get_ifd(0x8825)
    g[1], g[2] = ("N" if gps[0] >= 0 else "S"), dms(gps[0])
    g[3], g[4] = ("E" if gps[1] >= 0 else "W"), dms(gps[1])
    return ex.tobytes()


def make_vehicle_photos(inv: Invoice, folder: Path, split: str, seed: int, difficulty: str) -> list[dict]:
    hard = difficulty == "hard"
    rng = random.Random(f"{seed}-{inv.invoice_id}-photos-{difficulty}")
    vin = inv.vin or rng.choice(load_vin_pool()[(inv.model.make, inv.model.model)][inv.year])  # the car has one
    if inv.plate:
        plate, state = inv.plate, inv.plate_state
    else:  # the invoice didn't print a plate, but the car has one
        state = rng.choice(list(PLATE_FORMATS))
        plate = make_plate(rng, state)
    device = rng.choice(DEVICES)
    gps = shop_gps(inv.vendor)
    day = datetime.combine(inv.invoice_date - timedelta(days=rng.randint(0, 2)), datetime.min.time())
    taken = day + timedelta(hours=rng.randint(8, 16), minutes=rng.randint(0, 59))
    keeps_exif = rng.random() < 0.75  # messaging apps and some uploads strip EXIF
    uploaded = taken + timedelta(minutes=rng.randint(5, 240))
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, kind in enumerate(PHOTO_KINDS):
        when = taken + timedelta(minutes=i, seconds=rng.randint(0, 59))
        if kind == "sticker":
            img, _ = draw_sticker(inv, vin, rng, hard)
        elif kind == "vinplate":
            img = draw_vin_plate(inv, vin, rng, hard)
        elif kind == "plate":
            img = draw_plate(inv, plate, state, rng, hard)
        else:
            img = draw_odometer(inv, rng, hard)
        rel = f"{split}/photos/{inv.invoice_id}_{kind}.jpg"
        extra = {"exif": _exif_bytes(when, gps, device)} if keeps_exif else {}
        img.save(folder.parent / rel, "JPEG", quality=rng.randint(60, 75) if hard else rng.randint(78, 90), **extra)
        shot = PHOTO_KINDS[kind]
        rows.append({
            "photo": rel, "invoice_id": inv.invoice_id, "split": split, "shot_type": shot,
            "vin": vin if shot in ("door_jamb", "vin_plate") else "",
            "plate": plate if shot == "plate" else "", "plate_state": state if shot == "plate" else "",
            "odometer": inv.mileage if shot == "odometer" else "", "color": "", "body_style": "",
            "captured_at": (when if keeps_exif else uploaded).isoformat(), "uploaded_at": uploaded.isoformat(),
            "gps_lat": gps[0] if keeps_exif else "", "gps_lon": gps[1] if keeps_exif else "",
            "device": " ".join(device) if keeps_exif else "", "source": "synthetic",
        })
    return rows


def other_photo_row(inv: Invoice, split: str) -> dict:
    """An invoice photo, labeled shot type "other" (not a vehicle)."""
    row = dict.fromkeys(PHOTO_KEY_FIELDS, "")
    row.update(photo=f"{split}/{inv.invoice_id}_photo.jpg", invoice_id=inv.invoice_id, split=split,
               shot_type="other", source="synthetic")
    return row


# --- Matching scenario ------------------------------------------------------------
# Photo sets per dev vehicle, mixed into one stream, for tools/match_accuracy.py.

MATCH_KEY_FIELDS = [
    "photo", "set_id", "invoice_id", "case", "shot_type", "vin", "plate", "plate_state", "odometer", "color",
    "body_style", "captured_at", "exif", "uploaded_at", "uploader", "device", "gps_lat", "gps_lon", "source",
]
EXTERIORS_DIR = ROOT / "tests" / "fixtures" / "exteriors"


@dataclass
class PhotoSet:
    set_id: str
    vehicle: Invoice  # the car (for a car without an invoice: a stand-in with invoice_id "")
    invoice_id: str | None
    case: str
    vin: str
    plate: str
    state: str
    color: str | None
    shots: list[str]
    times: list[datetime]
    uploader: str
    device: tuple[str, str]
    exif: bool
    uploaded: datetime
    exterior: dict | None = None  # exteriors.csv row


def load_exteriors() -> list[dict]:
    path = EXTERIORS_DIR / "exteriors.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if (EXTERIORS_DIR / r["photo"]).exists()]


def plan_matching(invoices: list[Invoice], seed: int, exteriors: list[dict]) -> list[PhotoSet]:
    """Photo sets for the dev invoices' vehicles, with the hard cases:
    - invoice_without_photos: one invoice gets no set
    - lookalike_original / lookalike_no_invoice: a second car of the same model and color,
      no invoice, photographed at the same shop by the same person 12-20 minutes later
    - no_vin_or_plate: two sets with only exterior and odometer shots
    - no_exif_batch: two cars from one shop, EXIF stripped, uploaded together later
    - no_invoice: a car nobody invoiced
    - about a quarter of the other sets have no EXIF (upload time only)
    """
    rng = random.Random(f"{seed}-matching")
    pool = load_vin_pool()
    used_vins = {i.vin for i in invoices if i.vin}
    used_plates = {(i.plate, i.plate_state) for i in invoices if i.plate}
    by_color: dict[tuple, list[dict]] = {}
    by_model: dict[tuple, list[dict]] = {}
    for r in sorted(exteriors, key=lambda r: r["photo"]):
        by_color.setdefault((r["make"], r["model"], r["color"]), []).append(r)
        by_model.setdefault((r["make"], r["model"]), []).append(r)
    vendors = sorted({i.vendor for i in invoices})
    staff = {v: tuple(rng.sample(FIRST, 2)) for v in vendors}
    phone = {name: DEVICES[k % len(DEVICES)] for k, name in enumerate(sorted({n for t in staff.values() for n in t}))}

    def new_vin(make: str, model: str) -> tuple[str, int]:
        options = sorted((v, y) for y, vs in pool[(make, model)].items() for v in vs if v not in used_vins)
        vin, year = rng.choice(options)
        used_vins.add(vin)
        return vin, year

    def new_plate(state: str | None = None) -> tuple[str, str]:
        state = state or rng.choice(sorted(PLATE_FORMATS))
        while True:
            plate = make_plate(rng, state)
            if (plate, state) not in used_plates:
                used_plates.add((plate, state))
                return plate, state

    def pick_exterior(make, model, color, exclude=None) -> dict | None:
        rows = by_color.get((make, model, color), []) if color else by_model.get((make, model), [])
        rows = [r for r in rows if r is not exclude]
        return rng.choice(rows) if rows else None

    def walkaround(has_ext: bool, ids: bool) -> list[str]:
        shots = ["exterior"] if has_ext else []
        if ids:
            r = rng.random()
            vin_shots = ["vinplate"] if r < 0.45 else ["sticker"] if r < 0.85 else ["sticker", "vinplate"]
            shots += ["plate", *vin_shots]
        return shots + ["odometer"]

    def times_from(start: datetime, n: int) -> list[datetime]:
        out = [start]
        for _ in range(n - 1):
            out.append(out[-1] + timedelta(seconds=rng.randint(20, 100)))
        return out

    def intake(inv: Invoice) -> datetime:
        day = datetime.combine(inv.invoice_date - timedelta(days=rng.randint(0, 2)), datetime.min.time())
        return day + timedelta(hours=rng.randint(8, 15), minutes=rng.randint(0, 59))

    def build(vehicle: Invoice, invoice_id, case, start=None, uploader=None, ids=True, exif=None,
              color=None, exterior_exclude=None, vin=None, plate=None) -> PhotoSet:
        if vin is None:
            vin = vehicle.vin or new_vin(vehicle.model.make, vehicle.model.model)[0]
            used_vins.add(vin)
        plate, state = plate or ((vehicle.plate, vehicle.plate_state) if vehicle.plate else new_plate())
        used_plates.add((plate, state))
        color = color or vehicle.color
        ext = pick_exterior(vehicle.model.make, vehicle.model.model, color, exterior_exclude)
        color = color or (ext["color"] if ext else None)
        shots = walkaround(ext is not None, ids)
        times = times_from(start or intake(vehicle), len(shots))
        uploader = uploader or rng.choice(staff[vehicle.vendor])
        exif = (rng.random() < 0.75) if exif is None else exif
        uploaded = times[-1] + timedelta(minutes=rng.randint(5, 180))
        return PhotoSet("", vehicle, invoice_id, case, vin, plate, state, color, shots, times, uploader,
                        phone[uploader], exif, uploaded, ext)

    invs = sorted(invoices, key=lambda i: i.invoice_id)
    picks = invs[:]
    rng.shuffle(picks)
    no_photos = picks.pop()
    twins = [i for i in picks if i.color and len(by_color.get((i.model.make, i.model.model, i.color), [])) >= 2]
    original = (twins or [i for i in picks if i.color])[0]
    picks.remove(original)
    no_id = {picks.pop().invoice_id, picks.pop().invoice_id}
    batch_a = picks.pop()  # and the same shop's car with the nearest invoice date
    batch_b = min((i for i in picks if i.vendor == batch_a.vendor),
                  key=lambda i: (abs((i.invoice_date - batch_a.invoice_date).days), i.invoice_id))
    picks.remove(batch_b)

    sets: list[PhotoSet] = []
    for inv in invs:
        if inv is no_photos:
            continue
        if inv is original:
            orig = build(inv, inv.invoice_id, "lookalike_original", exif=True)
            sets.append(orig)
            vin, year = new_vin(inv.model.make, inv.model.model)
            plate, state = new_plate(orig.state)
            twin_car = replace(inv, invoice_id="", vin=vin, year=year, plate=plate, plate_state=state,
                               mileage=max(500, inv.mileage + rng.choice((-1, 1)) * rng.randint(5000, 40000)))
            start = orig.times[-1] + timedelta(minutes=rng.randint(12, 20))
            sets.append(build(twin_car, None, "lookalike_no_invoice", start=start, uploader=orig.uploader,
                              exif=True, color=orig.color, exterior_exclude=orig.exterior, vin=vin,
                              plate=(plate, state)))
        elif inv.invoice_id in no_id:
            sets.append(build(inv, inv.invoice_id, "no_vin_or_plate", ids=False))
        elif inv is batch_a or inv is batch_b:
            sets.append(build(inv, inv.invoice_id, "no_exif_batch", exif=False, uploader=staff[inv.vendor][0]))
        else:
            sets.append(build(inv, inv.invoice_id, "normal"))
    # The batch: both uploaded together, hours after the later of the two.
    a, b = [s for s in sets if s.case == "no_exif_batch"]
    a.uploaded = max(a.times[-1], b.times[-1]) + timedelta(minutes=rng.randint(30, 600))
    b.uploaded = a.uploaded + timedelta(minutes=rng.randint(2, 4))
    # A car nobody invoiced, at a random shop on a day with other work.
    make, model = rng.choice(sorted(by_model) or sorted(pool))
    cat = next(m for m in CATALOG if (m.make, m.model) == (make, model))
    host = rng.choice(invs)
    vin, year = new_vin(make, model)
    stray = replace(host, invoice_id="", model=cat, year=year, vin=vin, plate=None, plate_state=None, color=None,
                    mileage=rng.randint(8000, 120000))
    sets.append(build(stray, None, "no_invoice", vin=vin))
    for n, s in enumerate(sets, 1):
        s.set_id = f"v{n:02d}"
    return sets


def _exterior_image(row: dict, rng: random.Random) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(EXTERIORS_DIR / row["photo"])).convert("RGB")
    w, h = img.size
    box = (round(w * rng.uniform(0, 0.06)), round(h * rng.uniform(0, 0.06)),
           round(w * rng.uniform(0.94, 1)), round(h * rng.uniform(0.94, 1)))
    img = ImageEnhance.Brightness(img.crop(box)).enhance(rng.uniform(0.85, 1.1))
    img.thumbnail((1280, 1280))
    return img


def make_matching(sets: list[PhotoSet], out: Path, seed: int, difficulty: str) -> list[dict]:
    """Render the sets to ``out/dev/matching/`` and return matching_key.csv rows."""
    hard = difficulty == "hard"
    folder = out / "dev" / "matching"
    folder.mkdir(parents=True, exist_ok=True)
    kinds = {**PHOTO_KINDS, "exterior": "exterior"}
    rows = []
    for s in sets:
        rng = random.Random(f"{seed}-{s.set_id}-matching-{difficulty}")
        car = replace(s.vehicle, color=s.color)
        gps = shop_gps(s.vehicle.vendor)
        for k, (shot, when) in enumerate(zip(s.shots, s.times)):
            if shot == "exterior":
                img = _exterior_image(s.exterior, rng)
            elif shot == "sticker":
                img, _ = draw_sticker(car, s.vin, rng, hard)
            elif shot == "vinplate":
                img = draw_vin_plate(car, s.vin, rng, hard)
            elif shot == "plate":
                img = draw_plate(car, s.plate, s.state, rng, hard)
            else:
                img = draw_odometer(car, rng, hard)
            rel = f"dev/matching/{s.set_id}_{k + 1}_{shot}.jpg"
            uploaded = s.uploaded + timedelta(seconds=2 * k)
            extra = {"exif": _exif_bytes(when, gps, s.device)} if s.exif else {}
            img.save(out / rel, "JPEG", quality=rng.randint(60, 75) if hard else rng.randint(78, 90), **extra)
            kind = kinds[shot]
            rows.append({
                "photo": rel, "set_id": s.set_id, "invoice_id": s.invoice_id or "", "case": s.case,
                "shot_type": kind, "vin": s.vin if kind in ("door_jamb", "vin_plate") else "",
                "plate": s.plate if kind == "plate" else "", "plate_state": s.state if kind == "plate" else "",
                "odometer": s.vehicle.mileage if kind == "odometer" else "",
                "color": (s.color or "") if kind == "exterior" else "",
                "body_style": body_style(s.vehicle.model) if kind == "exterior" else "",
                "captured_at": when.isoformat(), "exif": "yes" if s.exif else "no",
                "uploaded_at": uploaded.isoformat(), "uploader": s.uploader,
                "device": " ".join(s.device) if s.exif else "",
                "gps_lat": gps[0] if s.exif else "", "gps_lon": gps[1] if s.exif else "",
                "source": f"commons:{s.exterior['photo']}" if shot == "exterior" else "synthetic",
            })
    with open(out / "matching_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MATCH_KEY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


# --- Main -----------------------------------------------------------------------

ANSWER_FIELDS = [
    "invoice_id", "split", "layout", "difficulty", "pdf", "scan", "photo",
    "vendor", "invoice_no", "invoice_date", "vin", "plate", "plate_state",
    "year", "make", "model", "trim", "color", "mileage",
    "parts_total", "labor_total", "tax", "total", "line_items",
]


def answer_row(inv: Invoice, split: str, difficulty: str) -> dict:
    rel = lambda suffix: f"{split}/{inv.invoice_id}{suffix}"  # noqa: E731
    printed_trim = inv.layout in (1, 2, 4)  # layouts 3 and 5 don't print trim
    return {
        "invoice_id": inv.invoice_id, "split": split, "layout": inv.layout, "difficulty": difficulty,
        "pdf": rel(".pdf"), "scan": rel("_scan.jpg"), "photo": rel("_photo.jpg"),
        "vendor": inv.vendor, "invoice_no": inv.invoice_no, "invoice_date": inv.invoice_date.isoformat(),
        "vin": inv.vin or "", "plate": inv.plate or "", "plate_state": inv.plate_state or "",
        "year": inv.year, "make": inv.model.make, "model": inv.model.model,
        "trim": inv.trim if printed_trim else "", "color": inv.color or "", "mileage": inv.mileage,
        "parts_total": f"{inv.parts_total:.2f}", "labor_total": f"{inv.labor_total:.2f}",
        "tax": f"{inv.tax:.2f}", "total": f"{inv.total:.2f}",
        "line_items": json.dumps([
            {"kind": i.kind, "description": i.description, "part_no": i.part_no,
             "qty": str(i.qty), "unit_price": f"{i.unit_price:.2f}", "amount": f"{i.amount:.2f}"}
            for i in inv.items
        ]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--difficulty", choices=("easy", "hard"), default="easy")
    ap.add_argument("--out", type=Path, help="default: tests/fixtures/synthetic[_hard]")
    ap.add_argument("--count", type=int, default=60, help="total invoices, spread evenly over the 5 layouts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-images", action="store_true", help="PDFs and answer key only (fast)")
    ap.add_argument("--no-photos", action="store_true", help="skip the vehicle photos")
    ap.add_argument("--matching-only", action="store_true",
                    help="only (re)write the matching scenario (dev/matching/, matching_key.csv)")
    args = ap.parse_args(argv)
    hard = args.difficulty == "hard"
    args.out = args.out or DEFAULT_OUT[args.difficulty]

    rng = random.Random(args.seed)
    pool, used = load_vin_pool(), set()
    rows, photo_rows, dev_invoices = [], [], []
    for n in range(1, args.count + 1):
        layout = (n - 1) % 5 + 1
        split = "dev" if layout in DEV_LAYOUTS else "holdout"
        inv = make_invoice(rng, n, layout, pool, used)
        if split == "dev":
            dev_invoices.append(inv)
        if args.matching_only:
            continue
        folder = args.out / split
        folder.mkdir(parents=True, exist_ok=True)
        pdf_path = folder / f"{inv.invoice_id}.pdf"
        hand = random.Random(f"{args.seed}-{inv.invoice_id}-hand") if hard else None
        LAYOUTS[layout](inv, hand).output(str(pdf_path))
        if not args.no_images:
            img_rng = random.Random(f"{args.seed}-{inv.invoice_id}-{args.difficulty}")
            make_scan(pdf_path, folder / f"{inv.invoice_id}_scan.jpg", img_rng, hard)
            make_photo(pdf_path, folder / f"{inv.invoice_id}_photo.jpg", img_rng, hard)
            if split == "dev" and not args.no_photos:
                photo_rows += make_vehicle_photos(inv, folder, split, args.seed, args.difficulty)
                if sum(r["shot_type"] == "other" for r in photo_rows) < OTHER_PHOTOS:
                    photo_rows.append(other_photo_row(inv, split))
        rows.append(answer_row(inv, split, args.difficulty))
        print(f"{inv.invoice_id}  layout {layout}  {split:7}  {inv.vin or '(no VIN)':17}  "
              f"{inv.year} {inv.model.make} {inv.model.model}")

    if not args.no_images and not args.no_photos:
        sets = plan_matching(dev_invoices, args.seed, load_exteriors())
        match_rows = make_matching(sets, args.out, args.seed, args.difficulty)
        print(f"matching scenario: {len(sets)} photo sets, {len(match_rows)} photos -> {args.out / 'dev' / 'matching'}")
    if args.matching_only:
        return
    with open(args.out / "answer_key.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANSWER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if photo_rows:
        with open(args.out / "photos_answer_key.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=PHOTO_KEY_FIELDS)
            writer.writeheader()
            writer.writerows(photo_rows)
    dev = sum(r["split"] == "dev" for r in rows)
    print(f"\n{len(rows)} {args.difficulty} invoices ({dev} dev, {len(rows) - dev} holdout) -> {args.out}")


if __name__ == "__main__":
    main()
