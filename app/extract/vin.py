"""VIN utilities: check digit, OCR-confusable repair, candidate search, vPIC decode.

Check digit
-----------
Position 9 is a check digit (0-9 or X) computed from the other 16 characters.
Every vehicle sold in North America carries one, wherever it was built. Vehicles
for other markets may not. The ``VIN_MARKET`` setting decides when it is expected:

* ``north_america`` (default): on every VIN.
* ``any``: only when the first character is 1-5 (built in North America). A
  JHM... or WBA... VIN may be a European-market car without one.

The caller can override either with ``expect_check_digit=True/False``. When not
expected we still compute it: a pass is VALID, a failure is UNVERIFIED (low
confidence, goes to review). It is never a reason to discard.

Repair rules
------------
Step 1 - forced fixes. I, O and Q can never appear in a VIN, so they are always
rewritten. These don't depend on the check digit.

    ======  =======
    Read    Becomes
    ======  =======
    I       1
    O       0
    Q       0
    ======  =======

Step 2 - confusable swaps. Only when the check digit is expected and the result of
step 1 fails it. At most ONE swap per VIN. A swap is kept only if the check digit
then passes.

    ======  =======
    Read    Try
    ======  =======
    B       8
    8       B
    S       5
    5       S
    ======  =======

No other substitutions are tried (no 2/Z, 6/G, D/0, ...). A random 17-character
string passes the check digit about 1 time in 11, so every extra swap adds false
"repairs".

If exactly one swap passes -> REPAIRED. If more than one passes -> AMBIGUOUS
(all of them listed, none chosen). If none passes -> UNVERIFIED.

Statuses and auto-linking
-------------------------
    ==========  =====================================================  ===============
    Status      Meaning                                                Auto-link alone
    ==========  =====================================================  ===============
    VALID       Passes as read                                         yes
    NORMALIZED  Passes after forced fixes only                         yes
    REPAIRED    Passes after one confusable swap                       no (*)
    UNVERIFIED  Fails the check digit, or check digit not expected     no (*)
                and fails
    AMBIGUOUS   Several swaps pass                                     no
    INVALID     Wrong length or characters that can't be fixed         no
    ==========  =====================================================  ===============

(*) Allowed when a vPIC decode of that VIN agrees with the invoice
(``auto_link_eligible(vpic_agrees=True)``); otherwise a reviewer decides.
REPAIRED counts as unverified because a non-confusable misread elsewhere in the
VIN can make a wrong single swap pass the check digit by chance.

Assembled readings
------------------
``find_candidates`` also builds readings by joining tokens split across spaces or
a line break, or by cutting a 17-character window out of a longer run (VIN glued
to neighbouring text). Joining arbitrary codes on an invoice can produce a string
that passes the check digit by chance, so an assembled reading is only trusted
when a VIN label (VIN, V.I.N., Vehicle ID, Serial; any case) is on the same line
or the line above. Untrusted readings are kept, but their confidence is capped at
``UNLABELED_ASSEMBLY_CONFIDENCE`` and they never auto-link, even with vPIC.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from enum import StrEnum

import httpx

# --- Check digit tables -------------------------------------------------------

TRANSLITERATION: dict[str, int] = {
    **{str(d): d for d in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)

VIN_RE = re.compile(r"[A-HJ-NPR-Z0-9]{17}")

# Step 1 and step 2 of the repair rules in the module docstring.
FORCED_FIXES: dict[str, str] = {"I": "1", "O": "0", "Q": "0"}
CONFUSABLE_SWAPS: dict[str, str] = {"B": "8", "8": "B", "S": "5", "5": "S"}

# Pairs that count as "off by one OCR-confusable char" when comparing two VINs.
CONFUSABLE_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(p) for p in ("0O", "1I", "8B", "5S")
)

NORTH_AMERICA_WMI = "12345"
VIN_MARKETS = ("north_america", "any")
DEFAULT_VIN_MARKET = "north_america"


class VinStatus(StrEnum):
    VALID = "valid"
    NORMALIZED = "normalized"
    REPAIRED = "repaired"
    UNVERIFIED = "unverified"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


CONFIDENCE: dict[VinStatus, float] = {
    VinStatus.VALID: 0.95,
    VinStatus.NORMALIZED: 0.9,
    VinStatus.REPAIRED: 0.6,
    VinStatus.UNVERIFIED: 0.3,
    VinStatus.AMBIGUOUS: 0.2,
    VinStatus.INVALID: 0.0,
}
UNLABELED_ASSEMBLY_CONFIDENCE = 0.2


@dataclass(frozen=True)
class VinResult:
    raw: str
    vin: str | None  # best reading; None when AMBIGUOUS or INVALID
    status: VinStatus
    check_digit_expected: bool
    alternatives: tuple[str, ...] = ()  # every passing reading when AMBIGUOUS
    notes: tuple[str, ...] = ()
    unlabeled_assembly: bool = False  # see "Assembled readings" in the module docstring

    @property
    def confidence(self) -> float:
        if self.unlabeled_assembly:
            return min(CONFIDENCE[self.status], UNLABELED_ASSEMBLY_CONFIDENCE)
        return CONFIDENCE[self.status]

    def auto_link_eligible(self, vpic_agrees: bool = False) -> bool:
        """Can this reading auto-link on its own (see module docstring)?"""
        if self.unlabeled_assembly:
            return False
        if self.status in (VinStatus.VALID, VinStatus.NORMALIZED):
            return True
        if self.status in (VinStatus.REPAIRED, VinStatus.UNVERIFIED):
            return vpic_agrees
        return False


def normalize(raw: str) -> str:
    """Uppercase and drop whitespace, dashes and dots. Other characters are kept so
    validation can reject them."""
    return re.sub(r"[\s.\-]", "", raw).upper()


def compute_check_digit(vin: str) -> str:
    if len(vin) != 17 or any(c not in TRANSLITERATION for c in vin):
        raise ValueError(f"not a 17-character VIN alphabet string: {vin!r}")
    total = sum(TRANSLITERATION[c] * w for c, w in zip(vin, WEIGHTS))
    remainder = total % 11
    return "X" if remainder == 10 else str(remainder)


def validation_errors(vin: str) -> list[str]:
    """Strict checks on an already-normalized VIN. Empty list means valid."""
    if len(vin) != 17:
        return [f"length is {len(vin)}, expected 17"]
    bad = sorted({c for c in vin if c not in TRANSLITERATION})
    if bad:
        return [f"invalid character(s): {', '.join(bad)}"]
    expected = compute_check_digit(vin)
    if vin[8] != expected:
        return [f"check digit is {vin[8]}, expected {expected}"]
    return []


def is_valid(vin: str) -> bool:
    return not validation_errors(vin)


def vin_market() -> str:
    """The ``VIN_MARKET`` setting. A typo raises rather than silently changing
    which VINs get repaired."""
    market = os.getenv("VIN_MARKET", "").strip().lower() or DEFAULT_VIN_MARKET
    if market not in VIN_MARKETS:
        raise ValueError(f"VIN_MARKET must be one of {VIN_MARKETS}, got {market!r}")
    return market


def check_digit_expected(vin: str, market: str | None = None) -> bool:
    market = market or vin_market()
    if market not in VIN_MARKETS:
        raise ValueError(f"market must be one of {VIN_MARKETS}, got {market!r}")
    if market == "north_america":
        return bool(vin)
    return bool(vin) and vin[0] in NORTH_AMERICA_WMI


def repair(raw: str, expect_check_digit: bool | None = None) -> VinResult:
    """Apply the repair rules in the module docstring to one candidate reading."""
    s = normalize(raw)
    fixable = all(c in TRANSLITERATION or c in FORCED_FIXES for c in s)
    if len(s) != 17 or not fixable:
        return VinResult(
            raw=raw, vin=None, status=VinStatus.INVALID, check_digit_expected=False,
            notes=tuple(validation_errors(s)),
        )

    fixed = "".join(FORCED_FIXES.get(c, c) for c in s)
    expected = check_digit_expected(fixed) if expect_check_digit is None else expect_check_digit

    def result(status: VinStatus, vin: str | None, **kw) -> VinResult:
        return VinResult(raw=raw, vin=vin, status=status, check_digit_expected=expected, **kw)

    if is_valid(fixed):
        return result(VinStatus.VALID if fixed == s else VinStatus.NORMALIZED, fixed)
    if not expected:
        return result(VinStatus.UNVERIFIED, fixed, notes=("check digit not expected for this WMI",))

    passing = []
    for i, c in enumerate(fixed):
        if c in CONFUSABLE_SWAPS:
            candidate = fixed[:i] + CONFUSABLE_SWAPS[c] + fixed[i + 1:]
            if is_valid(candidate):
                passing.append(candidate)
    if len(passing) == 1:
        return result(VinStatus.REPAIRED, passing[0])
    if len(passing) > 1:
        return result(VinStatus.AMBIGUOUS, None, alternatives=tuple(passing))
    return result(VinStatus.UNVERIFIED, fixed, notes=tuple(validation_errors(fixed)))


# --- Finding candidates in text ------------------------------------------------

TOKEN_RE = re.compile(r"[A-Z0-9]+")
# Tokens may be joined across spaces/tabs and at most one line break.
JOIN_GAP_RE = re.compile(r"[ \t]*(?:\r?\n)?[ \t]*")
MAX_JOIN_TOKENS = 3
MAX_JOINED_LEN = 40


def _vin_shaped(s: str) -> bool:
    return any(c.isalpha() for c in s) and any(c.isdigit() for c in s)


def _passes_strictly(s: str) -> bool:
    """Gate for readings we assembled ourselves (joined tokens, windows of a longer
    run). No repair here: real VIN alphabet, serial ends in digits, check digit
    passes. Without this, joining arbitrary tokens would invent VINs."""
    return bool(VIN_RE.fullmatch(s)) and s[-3:].isdigit() and _vin_shaped(s) and is_valid(s)


# A VIN label on the candidate's line or the line above makes an assembled reading trusted.
LABEL_RE = re.compile(
    r"(?<![A-Z])(?:VIN|V\.\s?I\.\s?N|VEHICLE\s*ID(?:ENTIFICATION)?|SERIAL)(?![A-Z])"
)


@dataclass(frozen=True)
class Candidate:
    text: str
    start: int  # offsets into the original text
    end: int
    assembled: bool  # joined across whitespace or cut out of a longer run
    labeled: bool  # VIN label on the same line or the line above

    @property
    def trusted(self) -> bool:
        return not self.assembled or self.labeled


def _has_label_near(upper: str, pos: int) -> bool:
    line_start = upper.rfind("\n", 0, pos) + 1
    prev_start = upper.rfind("\n", 0, max(line_start - 1, 0)) + 1 if line_start else 0
    line_end = upper.find("\n", pos)
    region = upper[prev_start:line_end if line_end != -1 else len(upper)]
    return LABEL_RE.search(region) is not None


def find_candidates(text: str) -> list[Candidate]:
    """Return possible VIN readings in text order, deduplicated by text.

    Tiers, most trusted first:
      0. a whole token of exactly 17 characters (kept even if invalid; repair() decides)
      1. 2-3 tokens split by whitespace (at most one line break) that join to exactly 17
      2. a 17-window inside a longer token (VIN glued to neighbouring text)
      3. a 17-window across joined tokens (split and glued)
    Tiers 1-3 are "assembled": they must pass ``_passes_strictly`` and are only
    trusted with a VIN label nearby. A candidate is dropped if it overlaps one
    already found in a higher tier, so "VIN <vin> 84211" can't yield a shifted
    window that happens to pass the check digit.
    """
    upper = text.upper()
    tokens = list(TOKEN_RE.finditer(upper))
    found: list[tuple[int, int, int, str]] = []  # (tier, start, end, vin)

    for i, first in enumerate(tokens):
        joined = ""
        offsets: list[int] = []  # text offset of each character in `joined`
        for j in range(i, min(i + MAX_JOIN_TOKENS, len(tokens))):
            tok = tokens[j]
            if j > i and not JOIN_GAP_RE.fullmatch(upper[tokens[j - 1].end():tok.start()]):
                break
            joined += tok.group()
            offsets.extend(range(tok.start(), tok.end()))
            single = j == i

            if len(joined) == 17:
                if single and _vin_shaped(joined):
                    found.append((0, offsets[0], offsets[-1] + 1, joined))
                elif not single and _passes_strictly(joined):
                    found.append((1, offsets[0], offsets[-1] + 1, joined))
            elif len(joined) > 17:
                last_len = len(tok.group())
                for k in range(len(joined) - 16):
                    # Joined windows must start in the first token and end in the
                    # last; anything else is covered by a shorter join.
                    if not single and not (k < len(first.group()) and k + 17 > len(joined) - last_len):
                        continue
                    window = joined[k:k + 17]
                    if _passes_strictly(window):
                        found.append((2 if single else 3, offsets[k], offsets[k + 16] + 1, window))
            if len(joined) >= MAX_JOINED_LEN:
                break

    selected: list[tuple[int, int, int, str]] = []
    for tier in range(4):
        claimed = [(s, e) for _, s, e, _ in selected]
        for cand in (c for c in found if c[0] == tier):
            if not any(cand[1] < e and s < cand[2] for s, e in claimed):
                selected.append(cand)

    result: list[Candidate] = []
    for tier, start, end, vin in sorted(selected, key=lambda c: c[1]):
        cand = Candidate(vin, start, end, assembled=tier > 0, labeled=_has_label_near(upper, start))
        dup = next((n for n, c in enumerate(result) if c.text == vin), None)
        if dup is None:
            result.append(cand)
        elif cand.trusted and not result[dup].trusted:
            result[dup] = cand
    return result


def extract_vins(text: str) -> list[VinResult]:
    results: dict[str, VinResult] = {}
    for cand in find_candidates(text):
        r = repair(cand.text)
        if not cand.trusted:
            r = replace(
                r, unlabeled_assembly=True,
                notes=r.notes + ("assembled from split or glued text with no VIN label nearby",),
            )
        key = r.vin or r.raw
        if key not in results or (results[key].unlabeled_assembly and not r.unlabeled_assembly):
            results[key] = r
    return list(results.values())


def confusable_distance(a: str, b: str) -> int | None:
    """Number of differing positions if every difference is an OCR-confusable pair
    (0/O, 1/I, 8/B, 5/S); None if lengths differ or any difference isn't."""
    a, b = normalize(a), normalize(b)
    if len(a) != len(b):
        return None
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if any(frozenset((x, y)) not in CONFUSABLE_PAIRS for x, y in diffs):
        return None
    return len(diffs)


# --- vPIC ---------------------------------------------------------------------

DEFAULT_VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"


# vPIC "Error Code" values (GetVehicleVariableValuesList/Error Code). Informational
# codes describe the vehicle or the data NHTSA has; the VIN itself can still be right.
# Every other code (1-6, 11, 400, and any we don't know) means the VIN as given is
# likely wrong.
VPIC_INFORMATIONAL_CODES = {
    "0": "VIN decoded clean",
    "7": "manufacturer not registered with NHTSA",
    "8": "no detailed data available currently",
    "9": "glider warning",
    "10": "off-road vehicle warning",
    "12": "model year warning (only when a model year is sent with the request)",
    "14": "unable to provide information for some characters",  # see VpicDecode.partial
}


@dataclass(frozen=True)
class VpicDecode:
    vin: str
    ok: bool  # request succeeded and vPIC returned at least a make
    year: int | None = None
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    body_class: str | None = None
    error_codes: tuple[str, ...] = ()
    error_text: str = ""

    @property
    def problem_codes(self) -> tuple[str, ...]:
        return tuple(c for c in self.error_codes if c not in VPIC_INFORMATIONAL_CODES)

    @property
    def clean(self) -> bool:
        """Decoded with no code that says the VIN is wrong."""
        return self.ok and bool(self.error_codes) and not self.problem_codes

    @property
    def partial(self) -> bool:
        """Code 14: some characters couldn't be decoded. A misread in positions 4-8
        can look like this, so callers should ask for more agreement (e.g. model)."""
        return "14" in self.error_codes


_vpic_cache: dict[str, VpicDecode] = {}


def clear_vpic_cache() -> None:
    _vpic_cache.clear()


def _blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def decode_vpic(vin: str, client: httpx.Client | None = None) -> VpicDecode:
    """Decode a VIN with NHTSA vPIC. Never raises on network or response errors;
    returns ``ok=False`` with ``error_text`` instead. Only successes are cached."""
    vin = normalize(vin)
    if vin in _vpic_cache:
        return _vpic_cache[vin]

    base = os.getenv("VPIC_BASE_URL", DEFAULT_VPIC_BASE_URL).rstrip("/")
    url = f"{base}/DecodeVinValues/{vin}"
    params = {"format": "json"}
    try:
        if client is None:
            timeout = float(os.getenv("VPIC_TIMEOUT", "10"))
            with httpx.Client(timeout=timeout) as own_client:
                response = own_client.get(url, params=params)
        else:
            response = client.get(url, params=params)
        response.raise_for_status()
        row = response.json()["Results"][0]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
        return VpicDecode(vin=vin, ok=False, error_text=f"vPIC request failed: {type(e).__name__}: {e}")

    year = _blank_to_none(row.get("ModelYear"))
    make = _blank_to_none(row.get("Make"))
    decoded = VpicDecode(
        vin=vin,
        ok=make is not None,
        year=int(year) if year and year.isdigit() else None,
        make=make,
        model=_blank_to_none(row.get("Model")),
        trim=_blank_to_none(row.get("Trim")),
        body_class=_blank_to_none(row.get("BodyClass")),
        error_codes=tuple(c.strip() for c in str(row.get("ErrorCode", "")).split(",") if c.strip()),
        error_text=str(row.get("ErrorText", "")),
    )
    if decoded.ok:
        _vpic_cache[vin] = decoded
    return decoded


MAKE_ALIASES = {
    "CHEVY": "CHEVROLET",
    "VW": "VOLKSWAGEN",
    "MERCEDES": "MERCEDESBENZ",
    "MB": "MERCEDESBENZ",
}


def _squash(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _canonical_make(make: str) -> str:
    squashed = _squash(make)
    return MAKE_ALIASES.get(squashed, squashed)


def _models_agree(a: str, b: str) -> bool:
    # Invoices often append trim ("ACCORD EX", "F150 XLT"), so a prefix match agrees.
    a, b = _squash(a), _squash(b)
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def compare_with_invoice(
    decode: VpicDecode, *, year: int | None = None, make: str | None = None, model: str | None = None
) -> list[str]:
    """Fields where vPIC and the invoice disagree. Fields missing on either side are skipped."""
    problems = []
    if year is not None and decode.year is not None and int(year) != decode.year:
        problems.append(f"year: invoice {year}, vPIC {decode.year}")
    if make and decode.make and _canonical_make(make) != _canonical_make(decode.make):
        problems.append(f"make: invoice {make!r}, vPIC {decode.make!r}")
    if model and decode.model and not _models_agree(model, decode.model):
        problems.append(f"model: invoice {model!r}, vPIC {decode.model!r}")
    return problems


def agrees_with_invoice(
    decode: VpicDecode, *, year: int | None = None, make: str | None = None, model: str | None = None
) -> bool:
    """Positive agreement: decode succeeded, year and make present on both sides,
    and nothing conflicts. This is what lets an unverified VIN auto-link."""
    if not decode.ok or year is None or decode.year is None or not make or not decode.make:
        return False
    return not compare_with_invoice(decode, year=year, make=make, model=model)
