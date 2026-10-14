"""Scoring clusters against invoices and the link / review / unmatched decision."""

from datetime import date, datetime, timedelta

import pytest

from app.extract.invoice import FieldValue, InvoiceFields
from app.match import config as cfg
from app.match.cluster import PhotoObs, cluster_photos
from app.match.score import InvoiceObs, match, score_pair
from tests.test_invoice import HONDA, FakeVpic, make_vin

FORD = make_vin("1FTEW1EP_JF654321")
DAY = date(2026, 5, 4)
T0 = datetime(2026, 5, 4, 9, 0)


def cluster(*photos, uploader="ana", start=T0):
    """One cluster from photo specs (dicts), one minute apart."""
    obs = [PhotoObs(f"{uploader}{i}", uploader=uploader, time=start + timedelta(minutes=i),
                    time_from_exif=p.pop("exif", True), **p) for i, p in enumerate(photos)]
    [c] = cluster_photos(obs)
    c.id = f"c_{uploader}"
    return c


def vin_photo(vin=HONDA, trusted=True, source="barcode"):
    return {"vin": vin, "vin_trusted": trusted, "vin_source": source, "shot_type": "door_jamb"}


def plate_photo(plate="7KJD221", state="CA", trusted=True, source="alpr"):
    return {"plate": plate, "plate_state": state, "plate_trusted": trusted, "plate_source": source,
            "shot_type": "plate"}


def inv(iid="inv1", **kw):
    kw.setdefault("date", DAY)
    return InvoiceObs(iid, **kw)


def points(cand, name):
    return sum(s.points for s in cand.signals if s.name == name)


# --- Signals -------------------------------------------------------------------------


def test_trusted_vin_on_both_sides_is_a_hard_match():
    c = score_pair(cluster(vin_photo()), inv(vin=HONDA, vin_trusted=True))
    assert c.hard == "vin" and not c.veto


def test_vin_read_by_one_reader_scores_but_is_not_hard():
    c = score_pair(cluster(vin_photo(trusted=False, source="ocr:paddle")), inv(vin=HONDA, vin_trusted=True))
    assert c.hard is None and points(c, "vin") == cfg.VIN_ONE_READER
    c = score_pair(cluster(vin_photo()), inv(vin=HONDA, vin_trusted=False))
    assert c.hard is None and points(c, "vin") == cfg.VIN_ONE_READER


def test_model_only_vin_is_a_weak_signal():
    c = score_pair(cluster(vin_photo(trusted=False, source="vision:m")), inv(vin=HONDA, vin_trusted=True))
    assert c.hard is None and points(c, "vin") == cfg.MODEL_ONLY_ID
    c = score_pair(cluster(vin_photo()), inv(vin=HONDA, vin_trusted=False, vin_source="vision:m"))
    assert c.hard is None and points(c, "vin") == cfg.MODEL_ONLY_ID


def test_trusted_vin_conflict_is_a_veto():
    c = score_pair(cluster(vin_photo(FORD)), inv(vin=HONDA, vin_trusted=True))
    assert c.veto and "VIN" in c.veto


def test_untrusted_vin_disagreement_is_a_conflict_not_a_veto():
    c = score_pair(cluster(vin_photo(FORD, trusted=False, source="ocr:p")), inv(vin=HONDA, vin_trusted=True))
    assert not c.veto and any("VIN" in x for x in c.conflicts)


def test_vin_off_by_one_confusable_character():
    assert HONDA[5] == "8"
    b_for_8 = HONDA[:5] + "B" + HONDA[6:]  # an invoice OCR read that failed the check digit
    c = score_pair(cluster(vin_photo()), inv(vin=b_for_8, vin_trusted=False))
    assert points(c, "vin") == cfg.VIN_CONFUSABLE and c.hard is None
    seven_for_8 = HONDA[:5] + "7" + HONDA[6:]  # not a confusable pair
    c = score_pair(cluster(vin_photo()), inv(vin=seven_for_8, vin_trusted=False))
    assert points(c, "vin") == 0 and any("VIN" in x for x in c.conflicts)


def test_trusted_plate_and_state_is_a_hard_match():
    c = score_pair(cluster(plate_photo()), inv(plate="7KJD221", plate_state="CA", plate_trusted=True))
    assert c.hard == "plate" and points(c, "plate") == cfg.PLATE_STATE_EXACT


def test_plate_without_a_known_state_scores_but_is_not_hard():
    c = score_pair(cluster(plate_photo(state=None)), inv(plate="7KJD221", plate_state="CA", plate_trusted=True))
    assert c.hard is None and points(c, "plate") == cfg.PLATE_STATE_EXACT


def test_plate_in_another_state_is_a_conflict():
    c = score_pair(cluster(plate_photo(state="TX")), inv(plate="7KJD221", plate_state="CA", plate_trusted=True))
    assert c.hard is None and points(c, "plate") == 0 and any("state" in x for x in c.conflicts)


def test_untrusted_plate_matches_with_o_and_0_folded():
    c = score_pair(cluster(plate_photo("ODTR801", trusted=False)),
                   inv(plate="0DTR801", plate_state="CA", plate_trusted=True))
    assert c.hard is None and points(c, "plate") == cfg.PLATE_ONE_READER


def test_model_only_plates_are_weak_on_either_side():
    c = score_pair(cluster(plate_photo(trusted=False, source="vision:m")),
                   inv(plate="7KJD221", plate_state="CA", plate_trusted=True))
    assert points(c, "plate") == cfg.MODEL_ONLY_ID and c.hard is None
    c = score_pair(cluster(plate_photo()), inv(plate="7KJD221", plate_state="CA", plate_trusted=False))
    assert points(c, "plate") == cfg.MODEL_ONLY_ID and c.hard is None


def test_trusted_plate_conflict_is_a_veto():
    c = score_pair(cluster(plate_photo("8BWD418")), inv(plate="7KJD221", plate_state="CA", plate_trusted=True))
    assert c.veto and "plate" in c.veto


@pytest.mark.parametrize("days, exif, pts", [(0, True, 10), (1, True, 10), (-1, True, 10), (4, True, 5),
                                             (7, True, 0), (30, True, 0), (0, False, 5)])
def test_capture_date_near_invoice_date_decays(days, exif, pts):
    c = score_pair(cluster({"shot_type": "exterior", "exif": exif}, start=T0 - timedelta(days=days)), inv())
    assert points(c, "date") == pytest.approx(pts)


def test_odometer():
    odo = {"shot_type": "odometer", "odometer": 60513, "odometer_source": "ocr:p"}
    assert points(score_pair(cluster(dict(odo)), inv(mileage=60700)), "odometer") == cfg.ODOMETER
    c = score_pair(cluster(dict(odo)), inv(mileage=70000))
    assert points(c, "odometer") == 0 and any("odometer" in x for x in c.conflicts)
    model = dict(odo, odometer_source="vision:m")
    assert points(score_pair(cluster(model), inv(mileage=60513)), "odometer") == cfg.ODOMETER_MODEL_ONLY


def test_color_and_make_model():
    ext = {"shot_type": "exterior", "color": "silver"}
    assert points(score_pair(cluster(dict(ext)), inv(color="silver")), "color") == cfg.COLOR
    c = score_pair(cluster(dict(ext)), inv(color="red"))
    assert points(c, "color") == 0 and not c.veto
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    c = score_pair(cluster(vin_photo(trusted=False, source="ocr:p")), inv(make="Honda", model="Accord"), vpic=vpic)
    assert points(c, "make_model") == cfg.MAKE_MODEL


def test_invoice_obs_from_invoice_fields():
    f = InvoiceFields(fields={
        "vin": FieldValue(HONDA, 0.9, "", "pdf_text"), "plate": FieldValue("7KJD221", 0.9, "", "vision:m"),
        "plate_state": FieldValue("CA", 0.5, "", "vision:m"), "make": FieldValue("Honda", 0.9, "", "pdf_text"),
        "mileage": FieldValue(60513, 0.9, "", "pdf_text"), "invoice_date": FieldValue(DAY, 0.9, "", "pdf_text"),
        "color": FieldValue("silver", 0.9, "", "pdf_text"),
    }, vins=[], text_source="pdf_text")
    o = InvoiceObs.from_invoice("inv1", f)
    assert (o.vin, o.plate, o.plate_trusted, o.mileage, o.date, o.color) == (HONDA, "7KJD221", False, 60513, DAY,
                                                                              "silver")
    assert o.vin_trusted == f.vin_trusted


# --- Decisions -----------------------------------------------------------------------


def test_hard_match_auto_links():
    [d] = match([cluster(vin_photo())], [inv("inv1", vin=HONDA, vin_trusted=True), inv("inv2", vin=FORD)])
    assert (d.status, d.invoice_id, d.decided_by) == ("linked", "inv1", "vin")


def test_hard_match_with_a_conflict_goes_to_review():
    c = cluster(vin_photo(), plate_photo("8BWD418"))
    [d] = match([c], [inv("inv1", vin=HONDA, vin_trusted=True, plate="7KJD221", plate_state="CA",
                          plate_trusted=True)])
    assert d.status == "review" and any("plate" in r for r in d.reasons)


def test_score_auto_link_needs_80_and_a_25_point_lead():
    c = cluster(vin_photo(trusted=False, source="ocr:p"), {"shot_type": "odometer", "odometer": 1000,
                                                           "odometer_source": "ocr:p"})
    # 60 VIN + 15 odometer + 10 date = 85, next best 10 (date only)
    [d] = match([c], [inv("inv1", vin=HONDA, vin_trusted=True, mileage=1000), inv("inv2")])
    assert (d.status, d.invoice_id, d.decided_by) == ("linked", "inv1", "score") and d.score >= cfg.AUTO_SCORE


def test_close_second_sends_to_review():
    ext = {"shot_type": "exterior", "color": "silver"}
    odo = {"shot_type": "odometer", "odometer": 1000, "odometer_source": "ocr:p"}
    plate = plate_photo(trusted=False)
    [d] = match([cluster(ext, odo, plate)], [inv("inv1", plate="7KJD221", plate_state="CA", plate_trusted=True,
                                                 color="silver", mileage=1000),
                                             inv("inv2", color="silver", mileage=1200)])
    # inv1: 35 + 8 + 15 + 10 = 68; inv2: 8 + 15 + 10 = 33 -> review (below 80)
    assert d.status == "review" and [c.invoice_id for c in d.candidates][:2] == ["inv1", "inv2"]
    assert d.candidates[0].score == 68 and d.reasons


def test_weak_signals_alone_are_unmatched():
    [d] = match([cluster({"shot_type": "exterior", "color": "silver"})], [inv("inv1", color="silver")])
    assert d.status == "unmatched" and d.invoice_id is None and d.candidates


def test_vetoed_invoice_is_never_a_candidate():
    [d] = match([cluster(vin_photo(FORD))], [inv("inv1", vin=HONDA, vin_trusted=True)])
    assert d.status == "unmatched" and not d.candidates
    assert any("veto" in r for r in d.reasons)


def test_one_to_one_second_hard_match_goes_to_review():
    a, b = cluster(vin_photo()), cluster(vin_photo(), uploader="ben")
    d1, d2 = match([a, b], [inv("inv1", vin=HONDA, vin_trusted=True)])
    assert (d1.status, d1.invoice_id) == ("linked", "inv1")
    assert d2.status == "review" and any("already linked" in r for r in d2.reasons)


def test_linked_invoice_leaves_the_pool():
    a = cluster(vin_photo())
    b = cluster({"shot_type": "odometer", "odometer": 1000, "odometer_source": "ocr:p"},
                {"shot_type": "exterior", "color": "silver"}, uploader="ben")
    d1, d2 = match([a, b], [inv("inv1", vin=HONDA, vin_trusted=True, mileage=1000, color="silver"),
                            inv("inv2", mileage=5000, color="red")])
    assert (d1.status, d1.invoice_id) == ("linked", "inv1")
    assert d2.status != "linked" and all(c.invoice_id != "inv1" for c in d2.candidates)


def test_unassigned_photo_goes_to_review_without_scoring():
    obs = [PhotoObs("a", uploader="ana", time=T0, time_from_exif=True, vin=HONDA, vin_source="barcode"),
           PhotoObs("m", uploader="ana", time=T0 + timedelta(minutes=4), time_from_exif=True, shot_type="exterior"),
           PhotoObs("b", uploader="ana", time=T0 + timedelta(minutes=8), time_from_exif=True, vin=FORD,
                    vin_source="barcode")]
    decisions = match(cluster_photos(obs), [inv("inv1")])
    [d] = [d for d in decisions if d.status == "review" and "fits two clusters" in " ".join(d.reasons)]
    assert d.invoice_id is None


def test_decision_to_dict():
    [d] = match([cluster(vin_photo())], [inv("inv1", vin=HONDA, vin_trusted=True)])
    out = d.to_dict()
    assert out["status"] == "linked" and out["candidates"][0]["invoice_id"] == "inv1"
    assert out["candidates"][0]["signals"][0]["name"] == "vin"


# --- Likely misread invoice VIN or plate: review, never auto-link --------------------


def differ(vin: str, positions) -> str:
    """``vin`` with the characters at ``positions`` changed (check digit not fixed)."""
    out = list(vin)
    for i in positions:
        out[i] = "X" if out[i] != "X" else "Y"
    return "".join(out)


SILVER = {"shot_type": "exterior", "color": "silver"}


def test_invoice_vin_with_three_differences_goes_to_review_as_top_candidate():
    c = cluster(vin_photo(), dict(SILVER))
    [d] = match([c], [inv("inv1", vin=differ(HONDA, (1, 7, 15)), color="silver"),
                      inv("inv2", color="silver")])
    assert d.status == "review" and d.invoice_id is None
    assert d.candidates[0].invoice_id == "inv1"
    assert any("invoice VIN likely misread" in r for r in d.reasons)


def test_vin_misread_needs_make_model_or_color_agreement():
    vpic = FakeVpic(year=2003, make="HONDA", model="Accord")
    [d] = match([cluster(vin_photo())], [inv("inv1", vin=differ(HONDA, (2, 3)), make="Honda", model="Accord")],
                vpic=vpic)
    assert d.status == "review" and any("invoice VIN likely misread" in r for r in d.reasons)
    [d] = match([cluster(vin_photo(), {"shot_type": "exterior", "color": "red"})],
                [inv("inv1", vin=differ(HONDA, (2, 3)), color="silver")])
    assert d.status == "unmatched" and not any("misread" in r for r in d.reasons)


def test_vin_misread_only_for_a_trusted_photo_vin_and_an_untrusted_invoice_vin():
    near = differ(HONDA, (2, 3))
    [d] = match([cluster(vin_photo(trusted=False, source="ocr:p"), dict(SILVER))],
                [inv("inv1", vin=near, color="silver")])
    assert not any("misread" in r for r in d.reasons)
    [d] = match([cluster(vin_photo(), dict(SILVER))], [inv("inv1", vin=FORD, vin_trusted=True, color="silver")])
    assert d.status == "unmatched" and any("veto" in r for r in d.reasons)  # trusted on both sides: still a veto


def test_vin_with_four_differences_is_not_a_misread():
    [d] = match([cluster(vin_photo(), dict(SILVER))], [inv("inv1", vin=differ(HONDA, (1, 4, 7, 15)), color="silver")])
    assert d.status == "unmatched" and not any("misread" in r for r in d.reasons)


def test_misread_never_auto_links_even_with_a_high_score():
    # trusted plate (70) + color (8) + date (10) = 88, but the invoice VIN is 2 characters off
    c = cluster(vin_photo(), plate_photo(state=None), dict(SILVER))
    [d] = match([c], [inv("inv1", vin=differ(HONDA, (2, 3)), plate="7KJD221", plate_state="CA",
                          plate_trusted=True, color="silver"), inv("inv2")])
    assert d.status == "review" and d.candidates[0].invoice_id == "inv1"
    assert any("invoice VIN likely misread" in r for r in d.reasons)


def test_misread_candidate_must_be_unlinked():
    a = cluster(plate_photo())
    b = cluster(vin_photo(), dict(SILVER), uploader="ben")
    d1, d2 = match([a, b], [inv("inv1", plate="7KJD221", plate_state="CA", plate_trusted=True,
                                vin=differ(HONDA, (2, 3)), color="silver")])
    assert (d1.status, d1.invoice_id) == ("linked", "inv1")
    assert not any("misread" in r for r in d2.reasons) and all(x.invoice_id != "inv1" for x in d2.candidates)


@pytest.mark.parametrize("invoice_plate", ["EN90", "EN9060", "EN90804"])  # cut short, cut short, one off
def test_invoice_plate_prefix_or_near_match_goes_to_review_not_veto(invoice_plate):
    c = cluster(plate_photo("EN90604"), dict(SILVER))
    [d] = match([c], [inv("inv1", plate=invoice_plate, plate_state="CA", plate_trusted=True, color="silver")])
    assert d.status == "review" and d.candidates[0].invoice_id == "inv1"
    assert any("invoice plate likely misread" in r for r in d.reasons)
    assert d.candidates[0].veto is None


def test_plate_misread_needs_agreement_and_a_close_plate():
    [d] = match([cluster(plate_photo("EN90604"), {"shot_type": "exterior", "color": "red"})],
                [inv("inv1", plate="EN90", plate_state="CA", plate_trusted=True, color="silver")])
    assert d.status == "unmatched" and any("veto" in r for r in d.reasons)
    [d] = match([cluster(plate_photo("EN90604"), dict(SILVER))],
                [inv("inv1", plate="EN9", plate_state="CA", plate_trusted=True, color="silver")])
    assert not any("misread" in r for r in d.reasons)  # 3 characters: too short to call a prefix
    [d] = match([cluster(plate_photo("EN90604", trusted=False), dict(SILVER))],
                [inv("inv1", plate="EN90", plate_state="CA", plate_trusted=True, color="silver")])
    assert not any("misread" in r for r in d.reasons)  # the photo plate must be trusted
