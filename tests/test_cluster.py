"""Clustering photos into one cluster per vehicle (the clustering rules)."""

from datetime import datetime, timedelta

from app.extract.invoice import FieldValue
from app.extract.photo import PhotoFields
from app.match.cluster import PhotoObs, cluster_photos, fold_plate
from tests.test_invoice import HONDA, make_vin

FORD = make_vin("1FTEW1EP_JF654321")
T0 = datetime(2026, 5, 4, 9, 0)


def obs(pid, minute, uploader="ana", **kw):
    kw.setdefault("time_from_exif", True)
    return PhotoObs(pid, uploader=uploader, time=T0 + timedelta(minutes=minute), **kw)


def groups(clusters):
    return sorted(sorted(p.id for p in c.photos) for c in clusters if c.status == "cluster")


def unassigned(clusters):
    return sorted(p.id for c in clusters if c.status == "unassigned" for p in c.photos)


def test_fold_plate():
    assert fold_plate("odtr 801") == "0DTR801"
    assert fold_plate("7IKQ-221") == "71K0221"


def test_same_vin_or_plate_joins_photos_far_apart():
    photos = [obs("a", 0, vin=HONDA, vin_source="barcode", shot_type="door_jamb"),
              obs("b", 300, uploader="ben", vin=HONDA, vin_source="ocr:paddle", shot_type="vin_plate"),
              obs("c", 0, uploader="cy", plate="7KJD221", plate_source="alpr", shot_type="plate"),
              obs("d", 900, uploader="dee", plate="7KJD221", plate_source="alpr", shot_type="plate")]
    assert groups(cluster_photos(photos)) == [["a", "b"], ["c", "d"]]


def test_model_only_ids_and_exterior_plates_do_not_join():
    photos = [obs("a", 0, vin=HONDA, vin_source="vision:m", shot_type="vin_plate"),
              obs("b", 300, uploader="ben", vin=HONDA, vin_source="vision:m", shot_type="vin_plate"),
              obs("c", 0, uploader="cy", plate="MODEL3", plate_source="alpr", shot_type="exterior"),
              obs("d", 900, uploader="dee", plate="MODEL3", plate_source="alpr", shot_type="exterior")]
    assert groups(cluster_photos(photos)) == [["a"], ["b"], ["c"], ["d"]]


def test_same_uploader_within_ten_minutes_chains():
    photos = [obs("ext", 0, shot_type="exterior", color="red"), obs("vin", 2, vin=HONDA, vin_source="ocr:p"),
              obs("odo", 3, shot_type="odometer", odometer=1234), obs("plate", 11, plate="7KJD221",
                                                                      plate_source="alpr", shot_type="plate")]
    assert groups(cluster_photos(photos)) == [["ext", "odo", "plate", "vin"]]


def test_gap_over_ten_minutes_or_other_uploader_splits():
    photos = [obs("a1", 0, color="red", shot_type="exterior"), obs("a2", 1, shot_type="odometer"),
              obs("b1", 12, color="red", shot_type="exterior"),
              obs("c1", 1, uploader="ben", color="red", shot_type="exterior")]
    assert groups(cluster_photos(photos)) == [["a1", "a2"], ["b1"], ["c1"]]


def test_different_devices_do_not_chain():
    photos = [obs("a", 0, device="Apple iPhone 13"), obs("b", 2, device="Google Pixel 7"),
              obs("c", 4, device="Apple iPhone 13")]
    assert groups(cluster_photos(photos)) == [["a", "c"], ["b"]]


def test_color_or_body_conflict_blocks_time_link():
    photos = [obs("a", 0, color="red", shot_type="exterior"), obs("b", 2, color="blue", shot_type="exterior"),
              obs("c", 20, body_style="suv", shot_type="exterior"),
              obs("d", 22, body_style="sedan", shot_type="exterior")]
    assert groups(cluster_photos(photos)) == [["a"], ["b"], ["c"], ["d"]]


def test_vin_conflict_blocks_time_link():
    photos = [obs("a", 0, vin=HONDA, vin_source="barcode"), obs("b", 3, vin=FORD, vin_source="barcode")]
    assert groups(cluster_photos(photos)) == [["a"], ["b"]]


def test_photo_fitting_two_clusters_stays_unassigned():
    # Two cars by one uploader, 4 minutes apart: the exterior between them fits both.
    # a_odo is 8 minutes before b_vin but not its neighbor (mid_ext is): it stays with a.
    photos = [obs("a_vin", 0, vin=HONDA, vin_source="barcode"), obs("a_odo", 1, odometer=100),
              obs("mid_ext", 5, color="silver", shot_type="exterior"),
              obs("b_vin", 9, vin=FORD, vin_source="barcode"), obs("b_odo", 10, odometer=200)]
    cl = cluster_photos(photos)
    assert unassigned(cl) == ["mid_ext"]
    assert groups(cl) == [["a_odo", "a_vin"], ["b_odo", "b_vin"]]
    [u] = [c for c in cl if c.status == "unassigned"]
    assert "fits two clusters" in u.reason


def test_photo_with_its_own_id_fitting_two_clusters_is_its_own_cluster():
    # A plate shot between two cars: it can still match an invoice on its own plate.
    photos = [obs("a", 0, vin=HONDA, vin_source="barcode"), obs("p", 4, plate="7KJD221", plate_source="alpr",
                                                                  shot_type="plate"),
              obs("b", 8, vin=FORD, vin_source="barcode")]
    cl = cluster_photos(photos)
    assert not unassigned(cl)
    assert groups(cl) == [["a"], ["b"], ["p"]]
    [c] = [c for c in cl if [p.id for p in c.photos] == ["p"]]
    assert "fits two clusters" in c.reason


def test_upload_time_is_used_without_exif():
    photos = [obs("a", 0, time_from_exif=False), obs("b", 0, time_from_exif=False)]
    [c] = cluster_photos(photos)
    assert sorted(p.id for p in c.photos) == ["a", "b"] and not c.time_from_exif


def test_photo_without_time_or_uploader_is_alone():
    photos = [PhotoObs("a", uploader="ana"), PhotoObs("b", uploader="ana"), PhotoObs("c", time=T0),
              PhotoObs("d", time=T0)]
    assert groups(cluster_photos(photos)) == [["a"], ["b"], ["c"], ["d"]]


def test_cluster_summary():
    photos = [obs("ext", 0, color="red", body_style="suv", shot_type="exterior"),
              obs("vin", 1, vin=HONDA, vin_source="barcode", vin_trusted=True, shot_type="door_jamb"),
              obs("odo", 2, odometer=60513, odometer_source="ocr:paddle", shot_type="odometer")]
    [c] = cluster_photos(photos)
    assert (c.color, c.body_style, c.odometer, c.trusted_vin) == ("red", "suv", 60513, HONDA)
    assert c.time == T0 + timedelta(minutes=1) and c.time_from_exif
    assert c.id == "c001"


def test_from_photo_fields():
    pf = PhotoFields("x.jpg", fields={
        "shot_type": FieldValue("plate", 0.5, "", "vision:m"),
        "plate": FieldValue("7KJD221", 0.9, "", "alpr"), "plate_state": FieldValue("CA", 0.5, "", "vision:m"),
        "captured_at": FieldValue(T0, 0.9, "", "exif"), "device": FieldValue("Apple iPhone 13", 0.9, "", "exif"),
    }, plate_confirmed=True)
    o = PhotoObs.from_photo("x", pf, uploader="ana")
    assert (o.plate, o.plate_state, o.plate_trusted, o.plate_source) == ("7KJD221", "CA", True, "alpr")
    assert (o.time, o.time_from_exif, o.device, o.uploader, o.shot_type) == (T0, True, "Apple iPhone 13", "ana",
                                                                              "plate")
    pf.fields["captured_at"] = FieldValue(T0, 0.3, "", "upload")
    assert not PhotoObs.from_photo("x", pf).time_from_exif
