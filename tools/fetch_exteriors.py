"""Download a few freely licensed exterior photos per make/model/color from Wikimedia Commons.

The vehicles are the dev invoices' make, model and color (synthetic answer key). Photos
come from Commons' curated color categories, e.g. "Category:Red Toyota sedans", and
must have the model in the file title. The category gives the labels: color and body
style. Only CC0, public domain, CC BY and CC BY-SA files are kept.

Writes tests/fixtures/exteriors/<make>_<model>_<color>_<n>.jpg and exteriors.csv with
each photo's labels, source page, license and author. The images are gitignored
(re-download them with this script); exteriors.csv is committed so the attribution
travels with the repo.

Wikimedia's robot policy requires a User-Agent with contact details (requests without
are refused with 403): set WIKIMEDIA_CONTACT to your email or a URL.

Usage:
    python tools/fetch_exteriors.py [--per 2] [--difficulty hard] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from make_fixtures import BODY_STYLE, DEFAULT_OUT  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
OUT = ROOT / "tests" / "fixtures" / "exteriors"
FIELDS = ["photo", "shot_type", "make", "model", "color", "body_style", "title", "page_url", "image_url",
          "license", "license_url", "author", "credit", "category"]
PLURAL = {"sedan": "sedans", "suv": "SUVs", "pickup": "pickup trucks", "wagon": "station wagons"}
COLOR_WORDS = {"gray": ("Gray", "Grey")}  # Commons uses both spellings
FREE_LICENSE = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)?( \d\.\d)?)\b", re.I)
SKIP_TITLE = re.compile(r"interior|engine|badge|logo|emblem|dashboard|wheel|light|lamp|police|taxi|"
                        r"race|racing|nascar|livery|crash|wreck|model car|toy|scale", re.I)


def model_pattern(model: str) -> re.Pattern:
    """'CR-V' matches 'CR-V', 'CRV', 'CR V'; 'Silverado 1500' matches 'Silverado'."""
    core = model.split()[0] if model.startswith("Silverado") else model
    parts = [re.escape(p) for p in re.split(r"[-\s]", core)]
    return re.compile(r"(?<![A-Za-z0-9])" + r"[-\s]?".join(parts) + r"(?![A-Za-z0-9])", re.I)


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


class Commons:
    def __init__(self, contact: str) -> None:
        ua = f"vehicle-matcher-fixtures/0.1 (test fixtures for a portfolio project; {contact})"
        self.http = httpx.Client(headers={"User-Agent": ua}, timeout=60, follow_redirects=True)

    def query(self, **params) -> dict:
        r = self.http.get(API, params={"action": "query", "format": "json", "maxlag": 5, **params})
        r.raise_for_status()
        return r.json()

    def category_files(self, category: str, limit: int = 2000) -> list[str]:
        titles, cont = [], {}
        while len(titles) < limit:
            d = self.query(list="categorymembers", cmtitle=category, cmtype="file", cmlimit=500, **cont)
            titles += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
            if "continue" not in d:
                break
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        return titles

    def image_info(self, titles: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(titles), 50):
            d = self.query(titles="|".join(titles[i:i + 50]), prop="imageinfo",
                           iiprop="url|extmetadata|mime", iiurlwidth=1280)
            for page in d.get("query", {}).get("pages", {}).values():
                if page.get("imageinfo"):
                    out[page["title"]] = page["imageinfo"][0]
        return out


def vehicles(difficulty: str) -> list[tuple[str, str, str]]:
    key = DEFAULT_OUT[difficulty] / "answer_key.csv"
    with open(key, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "dev" and r["color"]]
    return sorted({(r["make"], r["model"], r["color"]) for r in rows})


def slug(*parts: str) -> str:
    return "_".join(re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-") for p in parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per", type=int, default=2, help="photos per make/model/color")
    ap.add_argument("--difficulty", choices=("easy", "hard"), default="hard", help="which answer key lists the vehicles")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    contact = os.getenv("WIKIMEDIA_CONTACT", "").strip()
    if not contact:
        print("Set WIKIMEDIA_CONTACT to your email or a URL: Wikimedia refuses API clients without contact details.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "exteriors.csv"
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            existing = {r["photo"]: r for r in csv.DictReader(fh)}
    commons = Commons(contact)
    rows = []
    for make, model, color in vehicles(args.difficulty):
        style = BODY_STYLE.get(model, "sedan")
        pattern = model_pattern(model)
        cands, category = [], ""
        for word in COLOR_WORDS.get(color, (color.capitalize(),)):
            for cat in (f"Category:{word} {make} {PLURAL[style]}", f"Category:{word} {make} automobiles"):
                files = [t for t in commons.category_files(cat) if pattern.search(t) and not SKIP_TITLE.search(t)]
                if files:
                    cands, category = files, cat
                    break
            if cands:
                break
        if not cands:
            print(f"  no photos: {color} {make} {model}")
            continue
        random.Random(f"{args.seed}-{make}-{model}-{color}").shuffle(cands)
        info = commons.image_info(cands[: 10 * args.per])
        got = 0
        for title in cands[: 10 * args.per]:
            ii = info.get(title)
            if got >= args.per or ii is None or ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            meta = ii.get("extmetadata", {})
            lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not FREE_LICENSE.match(lic):
                continue
            name = f"{slug(make, model, color)}_{got + 1}.jpg"
            dest = OUT / name
            if not dest.exists():
                r = commons.http.get(ii.get("thumburl") or ii["url"])
                r.raise_for_status()
                dest.write_bytes(r.content)
                time.sleep(0.5)  # be polite to upload.wikimedia.org
            rows.append({
                "photo": name, "shot_type": "exterior", "make": make, "model": model, "color": color,
                "body_style": style, "title": title, "page_url": ii.get("descriptionurl", ""),
                "image_url": ii.get("url", ""), "license": lic,
                "license_url": strip_html(meta.get("LicenseUrl", {}).get("value", "")),
                "author": strip_html(meta.get("Artist", {}).get("value", "")) or "unknown",
                "credit": strip_html(meta.get("Credit", {}).get("value", ""))[:200], "category": category,
            })
            got += 1
        print(f"{got} x {color} {make} {model}  ({category})")
    keep = {r["photo"] for r in rows}
    rows += [r for p, r in existing.items() if p not in keep and (OUT / p).exists()]
    rows.sort(key=lambda r: r["photo"])
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} photos -> {OUT}  (licenses and authors in {csv_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
