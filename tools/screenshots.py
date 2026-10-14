"""Screenshots of every page with Playwright: laptop (1440 px) and phone (390 px) width,
light and dark.

Needs a running app: ``uvicorn app.main:app --port 8765`` after ``tools/seed_demo.py``.
It answers one car in "Needs your check" (then undoes it) and processes the sample car
on the Upload page, so point it at a copy of the data or reset the demo afterwards.

The Find invoice by photo page is shot empty, with photos picked, with a confident
match, possible matches, and no match (with progress while the photos are read). Its
photos come from tests/fixtures (matching scenario, exteriors).

Usage:
    pip install playwright && python -m playwright install chromium
    python tools/screenshots.py [--url http://127.0.0.1:8765] [--out data/screenshots] [--no-upload]
                                [--only all|pages|find]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATCHING = ROOT / "tests" / "fixtures" / "synthetic_hard" / "dev" / "matching"
EXTERIORS = ROOT / "tests" / "fixtures" / "exteriors"
DEVICES = {  # name: (viewport, scale, mobile)
    "laptop": ({"width": 1440, "height": 900}, 1, False),
    "phone": ({"width": 390, "height": 844}, 3, True),
}
FIND_SETS = {
    "confident": sorted(MATCHING.glob("v03_*.jpg")),  # VIN plate + plate: inv_006
    "possible": sorted(MATCHING.glob("v12_*.jpg")),  # exterior + odometer only: inv_021
}
NO_MATCH = ["bmw_330i_blue_2.jpg", "honda_civic_blue_1.jpg", "jeep_grand-cherokee_black_2.jpg",
            "nissan_altima_white_1.jpg"]  # Commons exteriors, no date or ID: one per variant, read fresh
# (once cached, a lookup is too quick for the progress shot: pick unread ones then)


def shoot_find(browser, url: str, out: Path, shots: list, errors: list) -> None:
    k = 0
    for device, (viewport, scale, mobile) in DEVICES.items():
        for scheme in ("light", "dark"):
            page = browser.new_page(viewport=viewport, device_scale_factor=scale, is_mobile=mobile, has_touch=mobile,
                                    color_scheme=scheme)
            page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
            page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)

            def snap(name: str, full: bool = True) -> None:
                page.wait_for_timeout(300)
                path = out / f"{name}-{device}-{scheme}.png"
                page.screenshot(path=str(path), full_page=full)
                shots.append(path)

            def run(files, done=True):
                page.goto(f"{url}/find")
                page.set_input_files("#photos", [str(f) for f in files])
                page.click("#find-submit")
                page.wait_for_url(re.compile(r"/find/[^/]+$"))
                if done:
                    page.wait_for_selector("#find-status:not([hx-trigger])", timeout=180_000)
                    page.wait_for_load_state("networkidle")

            page.goto(f"{url}/find")
            page.wait_for_load_state("networkidle")
            snap("12-find")
            page.set_input_files("#photos", [str(f) for f in FIND_SETS["confident"]])
            snap("13-find-picked")
            run(FIND_SETS["confident"])
            snap("14-find-confident")
            run(FIND_SETS["possible"])
            snap("15-find-possible")
            run([EXTERIORS / NO_MATCH[k % len(NO_MATCH)]], done=False)
            k += 1
            try:  # a later step shows more than "opening the photos"
                page.wait_for_selector("text=/Now: (Looking at the photos|Reading the VIN|Reading the license plate)/", timeout=30_000)
                snap("16-find-progress", full=False)
            except Exception:
                pass  # finished before a step could be seen
            page.wait_for_selector("#find-status:not([hx-trigger])", timeout=180_000)
            page.wait_for_load_state("networkidle")
            snap("17-find-no-match")
            page.close()


def shoot_pages(browser, url: str, out: Path, shots: list, errors: list, upload: bool) -> None:
    job_url = None
    for device, (viewport, scale, mobile) in DEVICES.items():
        for scheme in ("light", "dark"):
            page = browser.new_page(viewport=viewport, device_scale_factor=scale, is_mobile=mobile, has_touch=mobile,
                                    color_scheme=scheme)
            page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
            first_variant = device == "laptop" and scheme == "light"

            def snap(name: str, full: bool = True) -> None:
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(300)
                path = out / f"{name}-{device}-{scheme}.png"
                page.screenshot(path=str(path), full_page=full)
                shots.append(path)

            page.goto(f"{url}/")
            snap("01-home")
            page.goto(f"{url}/vehicles")
            snap("02-vehicles")
            first = page.locator("a[href^='/vehicles/inv_']").first.get_attribute("href")
            page.goto(f"{url}{first}")
            snap("03-vehicle")
            # A scanned invoice, with "Show where each field was read" ticked
            hrefs = page.goto(f"{url}/vehicles") and page.locator("a[href^='/vehicles/inv_']").evaluate_all(
                "els => els.map(e => e.getAttribute('href'))")
            for href in hrefs:
                page.goto(f"{url}{href}")
                if page.locator(".field-box.src-ocr").count() >= 5:
                    page.get_by_text("Show where each field was read").click()
                    snap("04-vehicle-scan")
                    break
            page.goto(f"{url}/vehicles?show=no_invoice")
            snap("05-photos-without-invoice")
            page.goto(f"{url}/review")
            snap("06-needs-your-check")
            if first_variant:  # an answer, then the Undo banner on the next car
                page.locator("button[data-hotkey~='n']").click()
                page.wait_for_url(re.compile(r"done="))
                page.evaluate("window.scrollTo(0, 0)")
                snap("06b-needs-your-check-undo", full=False)
                page.locator("button[data-hotkey='u']").click()
                page.wait_for_load_state("networkidle")
            for _ in range(15):  # the car whose invoice VIN was probably misread
                if page.get_by_text("Probably misread on the invoice").count():
                    snap("07-needs-your-check-misread")
                    break
                nxt = page.locator("a[data-hotkey~='arrowright']")
                if not nxt.count():
                    break
                page.goto(url + nxt.get_attribute("href"))  # a tap can count twice on the phone emulation
            page.goto(f"{url}/results")
            snap("08-how-well-it-works")
            page.goto(f"{url}/upload")
            snap("09-upload")
            if job_url is None and upload and page.get_by_text("Process the sample car").count():
                page.get_by_text("Process the sample car").click()
                page.wait_for_url(re.compile(r"/upload/\d+$"))
                job_url = page.url
                try:
                    page.wait_for_selector("text=/Reading|Looking|Finding/", timeout=10_000)
                    snap("10-upload-progress", full=False)
                except Exception:
                    pass
                page.wait_for_selector("text=All steps finished", timeout=600_000)
                snap("11-upload-done")
            elif job_url:
                page.goto(job_url)
                snap("11-upload-done")
            page.close()


def main(argv: list[str] | None = None) -> int:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(ROOT / "data" / "screenshots"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--only", choices=("all", "pages", "find"), default="all")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots: list[Path] = []
    errors: list[str] = []

    with sync_playwright() as pw:
        # --use-mock-keychain: Chromium on macOS must never ask for the login keychain
        browser = pw.chromium.launch(args=["--use-mock-keychain"])
        if args.only in ("all", "pages"):
            shoot_pages(browser, args.url, out, shots, errors, upload=not args.no_upload)
        if args.only in ("all", "find"):
            shoot_find(browser, args.url, out, shots, errors)
        browser.close()

    for p in shots:
        print(p)
    if errors:
        print("\nbrowser console errors:", *errors, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
