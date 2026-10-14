"""Tests for app/extract/vin.py.

Fixture VINs were checked with an independent check-digit script, not with vin.py.
"""

import httpx
import pytest

from app.extract.vin import (
    VinStatus,
    VpicDecode,
    agrees_with_invoice,
    check_digit_expected,
    compare_with_invoice,
    compute_check_digit,
    confusable_distance,
    decode_vpic,
    extract_vins,
    find_candidates,
    is_valid,
    normalize,
    repair,
    validation_errors,
)

# Valid, North American (check digit expected)
BUICK = "1M8GDM9AXKP042788"  # check digit X
HONDA = "1HGCM82633A004352"  # check digit 3
# Valid, built in Japan
HONDA_JP = "JHMCM56557C404453"
# European style, position 9 isn't a check digit
VW_EU = "WVWZZZ1JZXW000001"


# --- Check digit / validation ---------------------------------------------------


@pytest.mark.parametrize(
    "vin_, expected",
    [(BUICK, "X"), (HONDA, "3"), (HONDA_JP, "5"), ("11111111111111111", "1")],
)
def test_compute_check_digit(vin_, expected):
    assert compute_check_digit(vin_) == expected


@pytest.mark.parametrize("vin_", [BUICK, HONDA, HONDA_JP, "11111111111111111"])
def test_valid_vins(vin_):
    assert is_valid(vin_)
    assert validation_errors(vin_) == []


def test_single_char_change_breaks_check_digit():
    assert not is_valid(HONDA[:-1] + "3")


@pytest.mark.parametrize(
    "bad, reason",
    [
        (HONDA[:-1], "length"),
        (HONDA + "1", "length"),
        ("1HGCM8263OA004352", "character"),  # O
        ("1HGCM82633AI04352", "character"),  # I
        ("1HGCM82633AQ04352", "character"),  # Q
        ("1HGCM82633A00435*", "character"),
        ("1HGCM82634A004352", "check digit"),
    ],
)
def test_validation_errors(bad, reason):
    errors = validation_errors(bad)
    assert errors
    assert any(reason in e for e in errors)
    assert not is_valid(bad)


def test_normalize_strips_separators_and_uppercases():
    assert normalize(" 1hgcm-826.33a 004352 ") == HONDA
    assert is_valid(normalize("1hgcm82633a004352"))


WMI_CASES = [
    (BUICK, True), ("2T1BURHE0JC000000", True), ("3VWFE21C04M000001", True),
    (HONDA_JP, False), (VW_EU, False), ("KMHD84LF5HU000000", False),
]


@pytest.fixture
def any_market(monkeypatch):
    monkeypatch.setenv("VIN_MARKET", "any")


@pytest.mark.parametrize("vin_", [v for v, _ in WMI_CASES])
def test_north_america_market_is_default_and_expects_check_digit_everywhere(vin_):
    assert check_digit_expected(vin_) is True
    assert check_digit_expected(vin_, market="north_america") is True


@pytest.mark.parametrize("vin_, expected", WMI_CASES)
def test_any_market_uses_wmi_rule(vin_, expected, any_market):
    assert check_digit_expected(vin_) is expected


@pytest.mark.parametrize("vin_, expected", WMI_CASES)
def test_market_argument_overrides_env(vin_, expected):
    assert check_digit_expected(vin_, market="any") is expected


def test_market_setting_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("VIN_MARKET", " ANY ")
    assert check_digit_expected(VW_EU) is False


def test_unknown_market_raises(monkeypatch):
    monkeypatch.setenv("VIN_MARKET", "europe")
    with pytest.raises(ValueError, match="VIN_MARKET"):
        check_digit_expected(BUICK)
    with pytest.raises(ValueError):
        repair(BUICK)


# --- Repair ---------------------------------------------------------------------


def test_repair_valid_vin_unchanged():
    r = repair(BUICK)
    assert r.status is VinStatus.VALID
    assert r.vin == BUICK
    assert r.auto_link_eligible()


def test_repair_normalizes_input_first():
    assert repair("1hgcm-82633a004352").vin == HONDA


@pytest.mark.parametrize(
    "raw",
    ["1M8GDM9AXKPO42788",   # O -> 0
     "IM8GDM9AXKP042788",   # I -> 1
     "1M8GDM9AXKPQ42788"],  # Q -> 0
)
def test_repair_forced_fixes(raw):
    r = repair(raw)
    assert r.status is VinStatus.NORMALIZED
    assert r.vin == BUICK
    assert r.auto_link_eligible()


def test_repair_forced_fix_decides_check_digit_expectation():
    # Leading I becomes 1 -> North American -> swaps allowed on top.
    r = repair("IM8GDM9AXKP04278B")
    assert r.status is VinStatus.REPAIRED
    assert r.vin == BUICK


def test_repair_single_confusable_swap():
    r = repair("1M8GDM9AXKP04278B")  # trailing 8 read as B
    assert r.status is VinStatus.REPAIRED
    assert r.vin == BUICK
    assert r.check_digit_expected


def test_repaired_vin_needs_vpic_to_auto_link():
    r = repair("1M8GDM9AXKP04278B")
    assert not r.auto_link_eligible()
    assert r.auto_link_eligible(vpic_agrees=True)


def test_repair_ambiguous_when_two_swaps_pass():
    r = repair("1GS52B5B32S7S5212")
    assert r.status is VinStatus.AMBIGUOUS
    assert r.vin is None
    assert set(r.alternatives) == {"1GS52BSB32S7S5212", "1GS52B5832S7S5212"}
    assert not r.auto_link_eligible()
    assert not r.auto_link_eligible(vpic_agrees=True)


def test_repair_only_one_swap_allowed():
    # Two confusable errors (8->B twice): fixing either alone doesn't pass.
    r = repair("1M8GDM9AXKP042B8B")
    assert r.status is VinStatus.UNVERIFIED
    assert r.vin == "1M8GDM9AXKP042B8B"


def test_repair_unfixable_north_american():
    r = repair("1HGCM82633A004353")
    assert r.status is VinStatus.UNVERIFIED
    assert r.vin == "1HGCM82633A004353"
    assert r.check_digit_expected
    assert r.confidence < 0.5
    assert not r.auto_link_eligible()
    assert r.auto_link_eligible(vpic_agrees=True)


def test_repair_swaps_non_north_american_wmi_in_north_america_market():
    # A WBA VIN on a North American lot carries a check digit, so the swap is meaningful.
    r = repair("WBABCA815282BB18C")
    assert r.check_digit_expected
    assert r.status is VinStatus.REPAIRED
    assert r.vin == "W8ABCA815282BB18C"


def test_repair_no_swaps_when_check_digit_not_expected(any_market):
    # Fails the check digit; B->8 at position 2 *would* pass, but in the "any"
    # market a WBA (Germany) VIN may never have had a check digit.
    r = repair("WBABCA815282BB18C")
    assert r.status is VinStatus.UNVERIFIED
    assert r.vin == "WBABCA815282BB18C"
    assert not r.check_digit_expected
    assert r.alternatives == ()


def test_repair_caller_override_beats_market(any_market):
    assert repair("WBABCA815282BB18C", expect_check_digit=True).status is VinStatus.REPAIRED
    assert repair("1M8GDM9AXKP04278B", expect_check_digit=False).status is VinStatus.UNVERIFIED


@pytest.mark.parametrize("market", ["north_america", "any"])
def test_repair_european_vin_without_check_digit_goes_to_review(market, monkeypatch):
    monkeypatch.setenv("VIN_MARKET", market)
    r = repair(VW_EU)
    assert r.status is VinStatus.UNVERIFIED
    assert r.vin == VW_EU
    assert r.check_digit_expected is (market == "north_america")
    assert not r.auto_link_eligible()


def test_repair_non_north_american_vin_that_passes_is_valid():
    r = repair(HONDA_JP)
    assert r.status is VinStatus.VALID
    assert r.auto_link_eligible()


def test_repair_forced_fixes_still_apply_without_check_digit():
    r = repair("WVWZZZ1JZXWOOOOO1")
    assert r.vin == VW_EU
    assert r.status is VinStatus.UNVERIFIED


@pytest.mark.parametrize("raw", ["1HGCM82633A00435", "1HGCM82633A0043521", "1HGCM82633A00435*", ""])
def test_repair_invalid(raw):
    r = repair(raw)
    assert r.status is VinStatus.INVALID
    assert r.vin is None
    assert r.confidence == 0.0
    assert not r.auto_link_eligible(vpic_agrees=True)


def test_confidence_ordering():
    assert (
        repair(BUICK).confidence
        > repair("1M8GDM9AXKPO42788").confidence
        > repair("1M8GDM9AXKP04278B").confidence
        > repair("1HGCM82633A004353").confidence
        > repair("1GS52B5B32S7S5212").confidence
        > repair("short").confidence
    )


# --- Finding VINs in text -------------------------------------------------------


def texts(text):
    return [c.text for c in find_candidates(text)]


def test_find_vin_in_invoice_text():
    text = f"Invoice #10023\nVehicle: 2003 Honda Accord\nVIN: {HONDA}\nMileage: 84,211"
    assert texts(text) == [HONDA]


def test_find_multiple_vins_in_order_without_duplicates():
    text = f"VIN {BUICK} ... see also {HONDA}. Repeat: {BUICK}"
    assert texts(text) == [BUICK, HONDA]


def test_find_lowercase_and_ocr_letters_kept_for_repair():
    text = "vin: 1m8gdm9axkpo42788"
    assert texts(text) == ["1M8GDM9AXKPO42788"]
    [result] = extract_vins(text)
    assert result.vin == BUICK
    assert result.status is VinStatus.NORMALIZED


def test_find_vin_without_check_digit_as_whole_token():
    assert texts(f"Fahrgestellnummer {VW_EU}") == [VW_EU]


@pytest.mark.parametrize(
    "text",
    [
        "1M8GDM9A XKP042788",
        "VIN: 1M8GDM9A\nXKP042788",
        "VIN: 1M8GDM9A \n  XKP042788 odometer 12,000",
        "1M8GD M9AXK P042788",  # three pieces
    ],
)
def test_find_vin_split_by_space_or_line_break(text):
    assert BUICK in texts(text)


def test_split_reading_must_pass_check_digit():
    # Two unrelated tokens whose lengths add up to 17.
    text = "PO ABC12345 DEFG67891 net 30"
    assert compute_check_digit("ABC12345DEFG67891") != "D"
    assert texts(text) == []


def test_split_not_joined_across_blank_line():
    assert texts("1M8GDM9A\n\nXKP042788") == []


@pytest.mark.parametrize(
    "text",
    [
        f"VIN{BUICK}MILES 12000",
        f"VIN#:{BUICK}MI",
        f"STOCK{BUICK}",
        f"{BUICK}ODO",
    ],
)
def test_find_vin_glued_to_neighbouring_text(text):
    assert texts(text) == [BUICK]


def test_find_vin_split_and_glued():
    assert texts("VIN1M8GDM9A XKP042788") == [BUICK]


# --- Labels on assembled readings -----------------------------------------------


def only_result(text):
    [r] = extract_vins(text)
    return r


def test_whole_token_trusted_without_label():
    r = only_result(f"Customer car {BUICK} in bay 3")
    assert not r.unlabeled_assembly
    assert r.auto_link_eligible()


@pytest.mark.parametrize(
    "text",
    [
        "1M8GDM9A XKP042788",
        "Notes:\n1M8GDM9A\nXKP042788",
        f"STOCK{BUICK}",
        f"{BUICK}ODO",
        "1M8GD M9AXK P042788",
    ],
)
def test_unlabeled_assembled_reading_is_low_confidence_and_never_auto_links(text):
    [cand] = find_candidates(text)
    assert cand.assembled and not cand.labeled and not cand.trusted
    r = only_result(text)
    assert r.vin == BUICK and r.status is VinStatus.VALID  # the reading itself is fine
    assert r.unlabeled_assembly
    assert r.confidence <= 0.2
    assert not r.auto_link_eligible()
    assert not r.auto_link_eligible(vpic_agrees=True)


@pytest.mark.parametrize(
    "text",
    [
        "VIN: 1M8GDM9A XKP042788",
        "vin 1M8GDM9A XKP042788",
        "V.I.N. 1M8GDM9A XKP042788",
        "V.I.N 1M8GDM9A XKP042788",
        "Vehicle ID: 1M8GDM9A XKP042788",
        "Vehicle Identification Number 1M8GDM9A XKP042788",
        "Serial No. 1M8GDM9A XKP042788",
        "1M8GDM9A XKP042788  (VIN)",              # label later on the same line
        "VIN NUMBER\n1M8GDM9A XKP042788",         # label on the line above
        "VIN:\n1M8GDM9A\nXKP042788",              # line above the start of a split
        f"VIN{BUICK}MILES 12000",                  # glued label
        "VIN1M8GDM9A XKP042788",
        f"Desc: tow-in\r\nVIN:\r\nSTOCK{BUICK}",
    ],
)
def test_labeled_assembled_reading_is_trusted(text):
    [cand] = find_candidates(text)
    assert cand.assembled and cand.labeled and cand.trusted
    r = only_result(text)
    assert not r.unlabeled_assembly
    assert r.confidence == 0.95
    assert r.auto_link_eligible()


@pytest.mark.parametrize(
    "text",
    [
        "VIN:\nRO 4411\n1M8GDM9A XKP042788",   # label two lines up
        "VINYL WRAP 1M8GDM9A XKP042788",          # not a label
        "DAVIN 1M8GDM9A XKP042788",
        "Vehicle: Buick 1M8GDM9A XKP042788",      # "Vehicle" alone isn't a VIN label
    ],
)
def test_label_must_be_a_vin_label_on_same_or_previous_line(text):
    [cand] = find_candidates(text)
    assert not cand.labeled
    assert only_result(text).unlabeled_assembly


def test_duplicate_prefers_trusted_reading():
    text = f"1M8GDM9A XKP042788\nVIN: {BUICK}"
    [cand] = find_candidates(text)
    assert cand.trusted
    assert not only_result(text).unlabeled_assembly


def test_unlabeled_note_explains_why():
    r = only_result("STOCK1M8GDM9AXKP042788")
    assert r.confidence == 0.2
    assert "no VIN label" in r.notes[-1]


@pytest.mark.parametrize(
    "text",
    [
        "Invoice number 12345 total $1,204.33",
        "INTERNATIONALISTS",               # 17 letters, no digits
        "12345678901234567",               # 17 digits, no letters
        "REFERENCECODEABCDEFGHJKLMN12345",  # long run, no window passes
    ],
)
def test_find_ignores_non_vins(text):
    assert texts(text) == []


def test_extract_vins_returns_results_with_status():
    text = f"VIN {HONDA}\nPrior VIN on file: 1HGCM82633A004353"
    results = extract_vins(text)
    assert [(r.vin, r.status) for r in results] == [
        (HONDA, VinStatus.VALID),
        ("1HGCM82633A004353", VinStatus.UNVERIFIED),
    ]


# --- Confusable distance --------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (BUICK, BUICK, 0),
        (BUICK, "1M8GDM9AXKPO42788", 1),   # 0/O
        (BUICK, "1MBGDM9AXKP042788", 1),   # 8/B
        ("1HGCM82633A004352", "1HGCM82633A0043S2", 1),  # 5/S
        ("1HGCM82633A004352", "IHGCM82633A004352", 1),  # 1/I
        (BUICK, "1MBGDM9AXKPO42788", 2),
        ("1HGCM82633A004352", "1HGCM82633A004353", None),  # 2 vs 3 isn't confusable
        (BUICK, BUICK[:-1], None),
    ],
)
def test_confusable_distance(a, b, expected):
    assert confusable_distance(a, b) == expected


# --- vPIC -----------------------------------------------------------------------


def vpic_payload(**overrides):
    result = {
        "ModelYear": "2003", "Make": "HONDA", "Model": "Accord", "Trim": "EX",
        "BodyClass": "Coupe", "ErrorCode": "0",
        "ErrorText": "0 - VIN decoded clean. Check Digit (9th position) is correct",
        "VIN": HONDA,
    }
    result.update(overrides)
    return {"Count": 1, "Message": "Results returned successfully", "Results": [result]}


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_decode_vpic_success():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=vpic_payload())

    d = decode_vpic(HONDA, client=mock_client(handler))
    assert d.ok
    assert (d.year, d.make, d.model, d.trim, d.body_class) == (2003, "HONDA", "Accord", "EX", "Coupe")
    assert d.error_codes == ("0",)
    assert seen[0].url.path.endswith(f"/DecodeVinValues/{HONDA}")
    assert seen[0].url.params["format"] == "json"


def test_decode_vpic_empty_fields_become_none():
    client = mock_client(lambda r: httpx.Response(200, json=vpic_payload(Trim="", BodyClass="")))
    d = decode_vpic(HONDA, client=client)
    assert d.trim is None and d.body_class is None


def test_decode_vpic_reports_vpic_error_codes():
    payload = vpic_payload(ErrorCode="1,11", ErrorText="1 - Check Digit (9th position) does not calculate properly")
    d = decode_vpic(HONDA, client=mock_client(lambda r: httpx.Response(200, json=payload)))
    assert d.ok  # still decoded a make
    assert d.error_codes == ("1", "11")
    assert "Check Digit" in d.error_text


def test_decode_vpic_no_make_is_not_ok():
    payload = vpic_payload(Make="", Model="", ModelYear="", ErrorCode="7", ErrorText="7 - Manufacturer is not registered")
    d = decode_vpic(VW_EU, client=mock_client(lambda r: httpx.Response(200, json=payload)))
    assert not d.ok
    assert d.year is None


def test_decode_vpic_timeout_returns_error_result():
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    d = decode_vpic(HONDA, client=mock_client(handler))
    assert not d.ok
    assert "timed out" in d.error_text.lower() or "timeout" in d.error_text.lower()


@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, text="<html>")])
def test_decode_vpic_bad_response_returns_error_result(response):
    d = decode_vpic(HONDA, client=mock_client(lambda r: response))
    assert not d.ok


def test_decode_vpic_caches_successes():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=vpic_payload())

    client = mock_client(handler)
    decode_vpic(HONDA, client=client)
    decode_vpic(HONDA.lower(), client=client)
    assert len(calls) == 1


def test_decode_vpic_does_not_cache_failures():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=vpic_payload())

    client = mock_client(handler)
    assert not decode_vpic(HONDA, client=client).ok
    assert decode_vpic(HONDA, client=client).ok
    assert len(calls) == 2


def test_decode_vpic_uses_base_url_from_env(monkeypatch):
    monkeypatch.setenv("VPIC_BASE_URL", "https://example.test/vpic/")
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json=vpic_payload())

    decode_vpic(HONDA, client=mock_client(handler))
    assert str(seen[0]).startswith(f"https://example.test/vpic/DecodeVinValues/{HONDA}")


# --- Invoice comparison ---------------------------------------------------------

DECODED = VpicDecode(vin=HONDA, ok=True, year=2003, make="HONDA", model="Accord")


def test_compare_with_invoice_match():
    assert compare_with_invoice(DECODED, year=2003, make="Honda", model="ACCORD EX") == []
    assert agrees_with_invoice(DECODED, year=2003, make="Honda", model="Accord")


def test_compare_with_invoice_make_mismatch():
    problems = compare_with_invoice(DECODED, year=2003, make="Toyota", model="Accord")
    assert len(problems) == 1 and "make" in problems[0]
    assert not agrees_with_invoice(DECODED, year=2003, make="Toyota", model="Accord")


def test_compare_with_invoice_year_mismatch():
    problems = compare_with_invoice(DECODED, year=2004, make="Honda")
    assert len(problems) == 1 and "year" in problems[0]


def test_compare_with_invoice_model_mismatch():
    assert any("model" in p for p in compare_with_invoice(DECODED, model="Civic"))


def test_compare_ignores_fields_missing_on_either_side():
    assert compare_with_invoice(DECODED) == []
    partial = VpicDecode(vin=HONDA, ok=True, make="HONDA")
    assert compare_with_invoice(partial, year=2003, make="Honda", model="Accord") == []


@pytest.mark.parametrize(
    "decoded_make, invoice_make, decoded_model, invoice_model",
    [
        ("CHEVROLET", "Chevy", "Silverado", "SILVERADO 1500"),
        ("VOLKSWAGEN", "VW", "Jetta", "jetta"),
        ("MERCEDES-BENZ", "Mercedes Benz", "C-Class", "C Class"),
        ("FORD", "Ford", "F-150", "F150 XLT"),
    ],
)
def test_compare_make_aliases_and_model_spelling(decoded_make, invoice_make, decoded_model, invoice_model):
    d = VpicDecode(vin=HONDA, ok=True, year=2020, make=decoded_make, model=decoded_model)
    assert compare_with_invoice(d, year=2020, make=invoice_make, model=invoice_model) == []


def test_agreement_needs_year_and_make():
    # Nothing to contradict isn't the same as agreeing: auto-link needs positive evidence.
    assert not agrees_with_invoice(DECODED, make="Honda")
    assert not agrees_with_invoice(DECODED, year=2003)
    assert not agrees_with_invoice(VpicDecode(vin=HONDA, ok=False), year=2003, make="Honda")


@pytest.mark.live
def test_decode_vpic_live():
    d = decode_vpic(HONDA)
    assert d.ok
    assert d.make == "HONDA"
    assert d.year == 2003


# --- vPIC error codes -------------------------------------------------------------


@pytest.mark.parametrize(
    "codes, clean",
    [
        (("0",), True),
        (("8",), True),     # no detailed data: informational
        (("7",), True),     # manufacturer not registered with NHTSA: informational
        (("0", "12"), True),  # model-year warning only applies when a year is sent
        (("9",), True), (("10",), True),
        (("14",), True),    # partial: informational, but see partial_decode
        (("1",), False), (("2",), False), (("3",), False), (("4",), False),
        (("5", "14"), False), (("6",), False), (("11",), False), (("400",), False),
        (("99",), False),   # unknown codes count as problems
        ((), False),
    ],
)
def test_vpic_problem_codes(codes, clean):
    d = VpicDecode(vin=HONDA, ok=True, make="HONDA", error_codes=codes)
    assert d.clean is clean
    if codes and not clean:
        assert d.problem_codes


def test_vpic_partial_decode_flag():
    assert VpicDecode(vin=HONDA, ok=True, error_codes=("14",)).partial
    assert not VpicDecode(vin=HONDA, ok=True, error_codes=("0",)).partial


def test_failed_request_is_never_clean():
    assert not VpicDecode(vin=HONDA, ok=False, error_codes=("0",)).clean
