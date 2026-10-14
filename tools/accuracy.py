"""Field accuracy of app/extract/invoice.py against the synthetic answer key.

Runs on the dev split (layouts 1-3) by default. The holdout split is for the final
check only and needs --final; don't tune rules on its output.

For each field and each rendering (PDF / scan / photo):
  accuracy  = correct / invoices where the answer key has a value
  false     = invoices where the key is blank but something was extracted
Also reports how many invoices were flagged for review, how many VINs were trusted
(allowed to auto-link), and how many trusted VINs are wrong (target: 0).

--vision runs the rules on every file first, then stops the OCR worker and sends the
remaining gaps to the vision model in one session (the two never share memory), and
prints rules-only and rules+AI side by side.

Usage:
    python tools/accuracy.py [--difficulty easy|hard] [--key PATH] [-v] [--final]
                             [--vision [--model NAME]]
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extract.invoice import FIELD_NAMES, MAKES, extract_invoice, fill_gaps_with_vision, stop_ocr  # noqa: E402
from app.extract.vision import VisionModel  # noqa: E402
from app.extract.vision import from_env as vision_from_env  # noqa: E402

DEFAULT_KEYS = {
    "easy": ROOT / "tests" / "fixtures" / "synthetic" / "answer_key.csv",
    "hard": ROOT / "tests" / "fixtures" / "synthetic_hard" / "answer_key.csv",
}
CACHE_DIR = ROOT / "data" / "ocr_cache"
RENDERINGS = ("pdf", "scan", "photo")


def squash(s: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def canon_make(s: object) -> str:
    up = str(s).upper().strip()
    return squash(MAKES.get(up, up))


def same_invoice_no(expected: str, got: object) -> bool:
    # "RO 478081" vs "478081": a leading letters-only prefix is the document type, not the number.
    e, g = squash(expected), squash(got)
    if e == g:
        return True
    longer, shorter = (e, g) if len(e) > len(g) else (g, e)
    return bool(shorter) and longer.endswith(shorter) and longer[: -len(shorter)].isalpha()


def same_number(expected: str, got: object) -> bool:
    try:
        return Decimal(str(expected)) == Decimal(str(got))
    except InvalidOperation:
        return False


COMPARE = {
    "vin": lambda e, g: squash(e) == squash(g),
    "plate": lambda e, g: squash(e) == squash(g),
    "plate_state": lambda e, g: squash(e) == squash(g),
    "year": same_number,
    "make": lambda e, g: canon_make(e) == canon_make(g),
    "model": lambda e, g: squash(e) == squash(g),
    "color": lambda e, g: str(e).lower() == str(g).lower(),
    "mileage": same_number,
    "invoice_date": lambda e, g: str(e) == str(g),
    "vendor": lambda e, g: squash(e) == squash(g),
    "invoice_no": same_invoice_no,
    "total": same_number,
}


class Score:
    """Per-field, per-rendering tallies for one pipeline (rules-only or rules+AI)."""

    def __init__(self) -> None:
        self.correct = defaultdict(int)   # (field, rendering) -> n
        self.expected = defaultdict(int)
        self.false_fill = defaultdict(int)
        self.review = defaultdict(int)
        self.seconds = defaultdict(float)
        self.trusted = defaultdict(int)
        self.wrong_trusted = defaultdict(int)
        self.by_source = defaultdict(lambda: [0, 0])  # (source kind, rendering) -> [right, wrong]
        self.conflicts = defaultdict(int)  # rendering -> fields left for review (OCR vs model disagree)
        self.model_plates = defaultdict(lambda: [0, 0])  # rendering -> [right, wrong], never auto-linked
        self.mistakes = []

    def add(self, row: dict, rendering: str, result) -> None:
        self.review[rendering] += result.needs_review
        self.conflicts[rendering] += len(result.conflicts)
        if "plate" in result.fields and not result.plate_trusted:
            self.model_plates[rendering][squash(result.value("plate")) != squash(row["plate"])] += 1
        if result.vin_trusted:
            self.trusted[rendering] += 1
            if squash(result.value("vin")) != squash(row["vin"]):
                self.wrong_trusted[rendering] += 1
                self.mistakes.append((row["invoice_id"], rendering, "WRONG TRUSTED VIN", row["vin"], result.value("vin")))
        for name in FIELD_NAMES:
            want, got = row[name], result.value(name)
            key = (name, rendering)
            fv = result.fields.get(name)
            kind = fv.source.split(":")[0] if fv else None
            if want == "":
                if got is not None:
                    self.false_fill[key] += 1
                    self.by_source[(kind, rendering)][1] += 1
                    self.mistakes.append((row["invoice_id"], rendering, name, "(blank)", got, kind))
                continue
            self.expected[key] += 1
            if got is not None and COMPARE[name](want, got):
                self.correct[key] += 1
                self.by_source[(kind, rendering)][0] += 1
            else:
                if got is not None:
                    self.by_source[(kind, rendering)][1] += 1
                self.mistakes.append((row["invoice_id"], rendering, name, want, got, kind))

    def cell(self, name: str, r: str) -> str:
        k = (name, r)
        e, c = self.expected[k], self.correct[k]
        cell = f"{c}/{e} {100 * c / e:5.1f}%" if e else "      -"
        return cell + (f" +{self.false_fill[k]}F" if self.false_fill[k] else "")

    def total(self, r: str) -> tuple[int, int]:
        return sum(self.correct[(n, r)] for n in FIELD_NAMES), sum(self.expected[(n, r)] for n in FIELD_NAMES)


def print_table(score: Score, n: int) -> None:
    header = f"{'field':14}" + "".join(f"{r:>18}" for r in RENDERINGS)
    print(header)
    print("-" * len(header))
    for name in FIELD_NAMES:
        print(f"{name:14}" + "".join(f"{score.cell(name, r):>18}" for r in RENDERINGS))
    print("-" * len(header))
    print(f"{'all fields':14}" + "".join(
        f"{f'{c}/{e} {100 * c / e:5.1f}%':>18}" for c, e in (score.total(r) for r in RENDERINGS)))
    print(f"{'needs review':14}" + "".join(f"{f'{score.review[r]}/{n}':>18}" for r in RENDERINGS))
    print(f"{'VIN trusted':14}" + "".join(f"{f'{score.trusted[r]}/{n}':>18}" for r in RENDERINGS))
    print(f"{'wrong+trusted':14}" + "".join(f"{score.wrong_trusted[r]:>18}" for r in RENDERINGS))
    print(f"{'sec/invoice':14}" + "".join(f"{score.seconds[r] / n:>18.2f}" for r in RENDERINGS))


def print_side_by_side(rules: Score, ai: Score, n: int, model: str) -> None:
    print(f"\nRules-only vs rules + {model} (cells: rules -> rules+AI)\n")
    header = f"{'field':14}" + "".join(f"{r:>24}" for r in RENDERINGS)
    print(header)
    print("-" * len(header))

    def pct(c, e):
        return f"{100 * c / e:.1f}%" if e else "-"

    for name in FIELD_NAMES:
        cells = []
        for r in RENDERINGS:
            k = (name, r)
            e = rules.expected[k]
            cell = f"{pct(rules.correct[k], e)} -> {pct(ai.correct[k], e)}"
            if rules.false_fill[k] or ai.false_fill[k]:
                cell += f" ({rules.false_fill[k]}F/{ai.false_fill[k]}F)"
            cells.append(f"{cell:>24}")
        print(f"{name:14}" + "".join(cells))
    print("-" * len(header))
    rows = [("all fields", lambda s, r: pct(*s.total(r))),
            ("needs review", lambda s, r: f"{s.review[r]}"),
            ("VIN trusted", lambda s, r: f"{s.trusted[r]}"),
            ("wrong+trusted", lambda s, r: f"{s.wrong_trusted[r]}"),
            ("sec/invoice", lambda s, r: f"{s.seconds[r] / n:.1f}")]
    for label, fmt in rows:
        print(f"{label:14}" + "".join(f"{f'{fmt(rules, r)} -> {fmt(ai, r)}':>24}" for r in RENDERINGS))
    print("\nFields taken from the vision model (right / wrong):")
    for r in RENDERINGS:
        right, wrong = ai.by_source[("vision", r)]
        print(f"  {r:6} {right} right, {wrong} wrong")
    print("\nOCR vs model disagreements left for review (date, invoice no., total, mileage):")
    print("  " + ", ".join(f"{r} {ai.conflicts[r]}" for r in RENDERINGS))
    print("Model-only plates (kept, never auto-linked), right / wrong:")
    print("  " + ", ".join(f"{r} {ai.model_plates[r][0]}/{ai.model_plates[r][1]}" for r in RENDERINGS))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--difficulty", choices=("easy", "hard"), default="easy")
    ap.add_argument("--key", type=Path, help="answer key CSV (default: from --difficulty)")
    ap.add_argument("--final", action="store_true", help="score the HOLDOUT split (final check only)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every wrong field")
    ap.add_argument("--no-vpic", action="store_true", help="skip vPIC (OCR VINs are then never trusted)")
    ap.add_argument("--vision", action="store_true",
                    help="also run the vision model (OLLAMA_MODEL or --model) on the gaps and compare")
    ap.add_argument("--model", help="Ollama model for --vision (default: OLLAMA_MODEL)")
    args = ap.parse_args(argv)

    args.key = args.key or DEFAULT_KEYS[args.difficulty]
    split = "holdout" if args.final else "dev"
    with open(args.key, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]  # filter before anything else
    if not rows:
        print(f"no {split} rows in {args.key}")
        return 1
    model = None
    if args.vision:
        model = VisionModel(args.model) if args.model else vision_from_env()
        if model is None:
            print("--vision needs --model or OLLAMA_MODEL")
            return 1

    base = args.key.parent
    vpic = False if args.no_vpic else None
    rules = Score()
    results = {}
    for row in rows:
        for rendering in RENDERINGS:
            t0 = time.perf_counter()
            result = extract_invoice(base / row[rendering], cache_dir=CACHE_DIR, vpic=vpic)
            rules.seconds[rendering] += time.perf_counter() - t0
            rules.add(row, rendering, result)
            results[(row["invoice_id"], rendering)] = result

    ai = None
    if model is not None:
        # Never OCR and the vision model at once: stop the OCR worker, then one model session.
        stop_ocr()
        ai = Score()
        with model.session():
            for row in rows:
                for rendering in RENDERINGS:
                    t0 = time.perf_counter()
                    result = fill_gaps_with_vision(copy.deepcopy(results[(row["invoice_id"], rendering)]),
                                                   base / row[rendering], model, vpic=vpic)
                    ai.seconds[rendering] += rules.seconds[rendering] / len(rows) + time.perf_counter() - t0
                    ai.add(row, rendering, result)
                    if args.verbose:
                        print(f"  {row['invoice_id']} {rendering:5} {time.perf_counter() - t0:5.1f}s", flush=True)

    n = len(rows)
    print(f"\n{split.upper()} split, {args.key.parent.name}: {n} invoices x {len(RENDERINGS)} renderings\n")
    print_table(rules, n)
    if ai is not None:
        print(f"\nRules + vision model ({model.model}):\n")
        print_table(ai, n)
        print_side_by_side(rules, ai, n, model.model)
    print("\n+NF = answer key blank but a value was extracted (N invoices)."
          " OCR is cached in data/ocr_cache/, so timings after the first run are mostly parsing.")

    if args.verbose:
        for label, score in (("rules-only", rules), ("rules+AI", ai)):
            if score is None:
                continue
            print(f"\nMistakes, {label} (invoice, rendering, field, expected, got, source):")
            for m in score.mistakes:
                print("  ", *m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
