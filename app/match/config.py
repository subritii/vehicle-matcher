"""Clustering and matching weights and thresholds of the design rules, in one place.

Only trusted VINs and plates (see PhotoFields.vin_trusted / plate_trusted and
InvoiceFields.vin_trusted / plate_trusted) are hard identifiers. A VIN or plate read by
one reader only counts as a score signal, and one read only by the vision model as a
weak signal at most.
"""

from datetime import timedelta

# --- Clustering ------------------------------------------------------------------------
CLUSTER_WINDOW = timedelta(minutes=10)  # same uploader/device, consecutive photos

# --- Scoring (points) ------------------------------------------------------------------
VIN_CONFUSABLE = 60  # VIN off by one OCR-confusable character
VIN_ONE_READER = 60  # exact VIN, but one side read by a single reader (not trusted)
PLATE_STATE_EXACT = 70  # trusted plate on both sides, same state (or a state unknown)
PLATE_ONE_READER = 35  # exact plate (O/0, I/1 folded) not trusted on one side
MODEL_ONLY_ID = 10  # VIN or plate read only by the vision model: weak at most
MAKE_MODEL = 10
COLOR = 8
DATE_MAX = 10  # capture date within DATE_FULL_DAYS of the invoice date
DATE_FULL_DAYS = 1
DATE_ZERO_DAYS = 7  # decays linearly to 0 here
UPLOAD_TIME_FACTOR = 0.5  # capture time known only from the upload: lower weight
ODOMETER = 15
ODOMETER_MILES = 500
ODOMETER_MODEL_ONLY = 7  # odometer read only by the vision model

# --- Decisions -------------------------------------------------------------------------
AUTO_SCORE = 80
AUTO_LEAD = 25
REVIEW_MIN = 40
TOP_CANDIDATES = 3
LOOKUP_MIN = 15  # Find invoice from photo: weaker candidates are not shown

# Likely misread invoice VIN or plate: review with that invoice on top, never auto-link.
# Needs a trusted photo VIN/plate and make/model or color agreeing.
MISREAD_VIN_EDITS = 3  # untrusted invoice VIN at most this many edits from the photo VIN
MISREAD_PLATE_EDITS = 1  # invoice plate this close to the photo plate (O/0, I/1 folded)
MISREAD_PLATE_MIN_PREFIX = 4  # or a cut-short invoice plate of at least this many characters

VIN_CONFUSABLE_PAIRS = frozenset({frozenset(p) for p in ("0O", "1I", "8B", "5S", "2Z", "6G")})
