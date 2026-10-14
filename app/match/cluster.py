"""Group photos into one cluster per vehicle (the clustering rules).

1. Photos sharing a read VIN or plate -> same cluster. VINs from a barcode or OCR (check
   digit valid), plates read by fast-alpr on a plate close-up; never model-only values,
   never plates from exteriors (badges and dealer frames read as plates).
2. Otherwise consecutive photos from the same uploader (and the same phone, when EXIF
   says) at most 10 minutes apart -> same cluster, unless VIN, plate, color or body
   style disagree.
3. A photo that fits two clusters stays unassigned and goes to review; if it has its own
   VIN or plate it becomes a cluster of its own instead (it can still match an invoice).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.match.config import CLUSTER_WINDOW


def fold_plate(plate: str) -> str:
    """Plate text for comparison: O/Q -> 0 and I -> 1, which readers confuse."""
    p = re.sub(r"[^A-Z0-9]", "", plate.upper())
    return p.translate(str.maketrans("OQI", "001"))


def _model_only(source: str) -> bool:
    return source.startswith("vision")


@dataclass
class PhotoObs:
    """What matching needs from one photo: PhotoFields plus upload metadata."""

    id: str
    uploader: str | None = None
    time: datetime | None = None  # EXIF capture time, else upload time
    time_from_exif: bool = False
    device: str | None = None
    shot_type: str = "other"
    vin: str | None = None
    vin_source: str = ""
    vin_trusted: bool = False
    plate: str | None = None
    plate_state: str | None = None
    plate_source: str = ""
    plate_trusted: bool = False
    odometer: int | None = None
    odometer_source: str = ""
    color: str | None = None
    body_style: str | None = None
    gps: tuple[float, float] | None = None

    @classmethod
    def from_photo(cls, pid: str, pf, uploader: str | None = None) -> PhotoObs:
        """From app.extract.photo.PhotoFields."""
        f = pf.fields

        def src(name):
            return f[name].source if name in f else ""

        return cls(
            pid, uploader=uploader, time=pf.value("captured_at"), time_from_exif=src("captured_at") == "exif",
            device=pf.value("device"), shot_type=pf.value("shot_type") or "other",
            vin=pf.value("vin"), vin_source=src("vin"), vin_trusted=pf.vin_trusted,
            plate=pf.value("plate"), plate_state=pf.value("plate_state"), plate_source=src("plate"),
            plate_trusted=pf.plate_trusted, odometer=pf.value("odometer"), odometer_source=src("odometer"),
            color=pf.value("color"), body_style=pf.value("body_style"), gps=pf.value("gps"),
        )

    @property
    def id_vin(self) -> str | None:
        """VIN that may join clusters: read by a barcode or OCR, not only by the model."""
        return self.vin if self.vin and not _model_only(self.vin_source) else None

    @property
    def id_plate(self) -> str | None:
        """Plate that may join clusters: fast-alpr on a plate close-up."""
        if self.plate and self.plate_source == "alpr" and self.shot_type == "plate":
            return fold_plate(self.plate)
        return None


@dataclass
class Cluster:
    id: str
    photos: list[PhotoObs]
    status: str = "cluster"  # or "unassigned": a photo that fits two clusters
    reason: str = ""

    def _values(self, name: str) -> list:
        return [getattr(p, name) for p in self.photos if getattr(p, name) is not None]

    @property
    def vins(self) -> set[str]:
        return set(self._values("id_vin"))

    @property
    def plates(self) -> set[str]:
        return set(self._values("id_plate"))

    @property
    def trusted_vin(self) -> str | None:
        return next((p.vin for p in self.photos if p.vin_trusted), None)

    @property
    def color(self) -> str | None:
        return _most_common(self._values("color"))

    @property
    def body_style(self) -> str | None:
        return _most_common(self._values("body_style"))

    @property
    def odometer(self) -> int | None:
        """Odometer read by OCR if any, else by the model."""
        ocr = [p.odometer for p in self.photos if p.odometer is not None and not _model_only(p.odometer_source)]
        return _most_common(ocr) if ocr else _most_common(self._values("odometer"))

    @property
    def odometer_model_only(self) -> bool:
        return all(_model_only(p.odometer_source) for p in self.photos if p.odometer is not None)

    @property
    def time_from_exif(self) -> bool:
        return any(p.time_from_exif for p in self.photos if p.time)

    @property
    def time(self) -> datetime | None:
        """Median capture time, from EXIF photos if any."""
        times = sorted(p.time for p in self.photos if p.time and (p.time_from_exif or not self.time_from_exif))
        return times[(len(times) - 1) // 2] if times else None

    @property
    def uploaders(self) -> set[str]:
        return set(self._values("uploader"))


def _most_common(values: list):
    return max(sorted(set(values), key=str), key=values.count) if values else None


class _Groups:
    """Union-find over photo indexes, with the values that must not disagree."""

    KEYS = ("id_vin", "id_plate", "color", "body_style")

    def __init__(self, photos: list[PhotoObs]) -> None:
        self.parent = list(range(len(photos)))
        self.values = [{k: {getattr(p, k)} - {None} for k in self.KEYS} for p in photos]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def compatible(self, a: dict, b: dict) -> bool:
        return all(not (a[k] and b[k]) or a[k] & b[k] for k in self.KEYS)

    def union(self, i: int, j: int) -> bool:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return True
        if not self.compatible(self.values[ri], self.values[rj]):
            return False
        lo, hi = min(ri, rj), max(ri, rj)
        self.parent[hi] = lo
        self.values[lo] = {k: self.values[lo][k] | self.values[hi][k] for k in self.KEYS}
        return True


def _same_phone(a: PhotoObs, b: PhotoObs) -> bool:
    return a.device is None or b.device is None or a.device == b.device


def _neighbors(photos: list[PhotoObs], window: timedelta) -> list[tuple[int, int]]:
    """(i, j) for consecutive photos of one uploader (same phone) within ``window``."""
    pairs = []
    by_uploader: dict[str, list[int]] = {}
    for i, p in enumerate(photos):
        if p.uploader and p.time:
            by_uploader.setdefault(p.uploader, []).append(i)
    for idx in by_uploader.values():
        idx.sort(key=lambda i: (photos[i].time, i))
        for a, i in enumerate(idx):
            nxt = next((j for j in idx[a + 1:] if _same_phone(photos[i], photos[j])), None)
            if nxt is not None and photos[nxt].time - photos[i].time <= window:
                pairs.append((i, nxt))
    return pairs


def cluster_photos(photos: list[PhotoObs], window: timedelta = CLUSTER_WINDOW) -> list[Cluster]:
    g = _Groups(photos)
    # Rule 1: shared VIN or plate.
    first: dict[tuple[str, str], int] = {}
    for i, p in enumerate(photos):
        for key in (("vin", p.id_vin), ("plate", p.id_plate)):
            if key[1] is None:
                continue
            if key in first:
                g.union(first[key], i)
            else:
                first[key] = i
    # Rule 2: same uploader and phone, consecutive, within the window; closest pairs first.
    pairs = _neighbors(photos, window)
    pairs.sort(key=lambda ij: (photos[ij[1]].time - photos[ij[0]].time, photos[ij[0]].time, ij))
    for i, j in pairs:
        g.union(i, j)
    # Rule 3: a photo whose neighbors are in two clusters it would both fit.
    adjacent: dict[int, set[int]] = {}
    for i, j in pairs:
        adjacent.setdefault(i, set()).add(j)
        adjacent.setdefault(j, set()).add(i)
    members: dict[int, list[int]] = {}
    for i in range(len(photos)):
        members.setdefault(g.find(i), []).append(i)
    alone: dict[int, str] = {}
    for i, near in adjacent.items():
        mine = {k: {getattr(photos[i], k)} - {None} for k in _Groups.KEYS}
        fits = set()
        for j in near:
            root = g.find(j)
            others = [m for m in members[root] if m != i]
            if not others:
                continue
            values = {k: set().union(*({getattr(photos[m], k)} - {None} for m in others)) for k in _Groups.KEYS}
            if g.compatible(mine, values):
                fits.add(root)
        if len(fits) > 1:
            alone[i] = "fits two clusters (same uploader, minutes apart)"

    clusters: list[Cluster] = []
    seen: set[int] = set()
    order = sorted(range(len(photos)), key=lambda i: (photos[i].time or datetime.max, i))
    for i in order:
        if i in alone:
            p = photos[i]
            has_id = p.id_vin is not None or p.id_plate is not None
            clusters.append(Cluster("", [p], "cluster" if has_id else "unassigned", alone[i]))
            continue
        root = g.find(i)
        if root in seen:
            continue
        seen.add(root)
        group = [photos[m] for m in sorted(members[root], key=lambda m: (photos[m].time or datetime.max, m))
                 if m not in alone]
        clusters.append(Cluster("", group))
    for n, c in enumerate(clusters, 1):
        c.id = f"c{n:03d}"
    return clusters

