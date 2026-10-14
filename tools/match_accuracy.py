"""Clustering and matching accuracy on the dev matching scenario.

Invoices: the 36 hard dev invoices, each as one file (pdf, scan or photo in turn).
Photos: tests/fixtures/synthetic_hard/dev/matching/ (python tools/make_fixtures.py
--difficulty hard [--matching-only]), labels in matching_key.csv.

Every raw engine output (OCR, fast-alpr, the vision model, vPIC) comes from
data/extract_cache/ (invoice OCR from data/ocr_cache/); only cache misses run an
engine, so after the first run nothing is re-read. Rules, weights and matching re-run
every time.

Reports auto-link precision (target 100%), wrong auto-links (target 0), the share of
clusters auto-linked / sent to review / unmatched, and clustering accuracy.

Usage:
    python tools/match_accuracy.py [--model NAME | --no-vision] [--no-vpic] [-v]
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extract import cache as ch  # noqa: E402
from app.extract import photo as ph  # noqa: E402
from app.extract.invoice import extract_invoice, fill_gaps_with_vision, stop_ocr  # noqa: E402
from app.extract.vision import VisionModel  # noqa: E402
from app.extract.vision import from_env as vision_from_env  # noqa: E402
from app.match.cluster import PhotoObs, cluster_photos  # noqa: E402
from app.match.score import InvoiceObs, match  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_hard"
OCR_CACHE = ROOT / "data" / "ocr_cache"
EXTRACT_CACHE = ROOT / "data" / "extract_cache"
RENDERINGS = ("pdf", "scan", "photo")


def load(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def extract_invoices(rows: list[dict], vision, vpic) -> tuple[dict[str, InvoiceObs], dict[str, Path]]:
    """Each invoice as one file, pdf/scan/photo in turn (by invoice order)."""
    files = {r["invoice_id"]: FIXTURES / r[RENDERINGS[k % 3]] for k, r in enumerate(rows)}
    fields = {iid: extract_invoice(p, cache_dir=OCR_CACHE, vpic=vpic) for iid, p in files.items()}
    if vision is not None:
        stop_ocr()  # never OCR and the vision model at once
        with vision.session():
            fields = {iid: fill_gaps_with_vision(copy.deepcopy(f), files[iid], vision, vpic=vpic)
                      for iid, f in fields.items()}
    return {iid: InvoiceObs.from_invoice(iid, f) for iid, f in fields.items()}, files


def pair_scores(true_of: dict[str, str], groups: list[list[str]]) -> tuple[float, float]:
    """Pairwise precision and recall of a clustering (lists of photo ids)."""
    same_pred = {frozenset(p) for g in groups for p in combinations(g, 2)}
    by_true = defaultdict(list)
    for pid, t in true_of.items():
        by_true[t].append(pid)
    same_true = {frozenset(p) for g in by_true.values() for p in combinations(g, 2)}
    tp = len(same_pred & same_true)
    return (tp / len(same_pred) if same_pred else 1.0), (tp / len(same_true) if same_true else 1.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="Ollama vision model (default: OLLAMA_MODEL)")
    ap.add_argument("--no-vision", action="store_true", help="rules + OCR + barcode + fast-alpr only")
    ap.add_argument("--no-vpic", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every cluster's decision")
    args = ap.parse_args(argv)

    key_path = FIXTURES / "matching_key.csv"
    if not key_path.exists():
        print("no matching scenario; run tools/make_fixtures.py --difficulty hard --matching-only")
        return 1
    photos = load(key_path)
    invoices = [r for r in load(FIXTURES / "answer_key.csv") if r["split"] == "dev"]

    cache = ch.DiskCache(EXTRACT_CACHE)
    vpic = False if args.no_vpic else ch.cached_vpic(cache)
    vision = None
    if not args.no_vision:
        model = VisionModel(args.model) if args.model else vision_from_env()
        if model is None:
            print("no vision model: pass --model, set OLLAMA_MODEL, or use --no-vision")
            return 1
        vision = ch.CachedVision(model, cache)

    t0 = time.perf_counter()
    inv_obs, inv_files = extract_invoices(invoices, vision, vpic)
    engines = ph.Engines(ocr=ch.CachedOcr(ph._SharedOcr(), cache, name=os.getenv("OCR_ENGINE", "paddle") or "paddle"),
                         alpr=ch.CachedAlpr(ph.FastAlpr(), cache), vision=vision)
    paths = [FIXTURES / r["photo"] for r in photos]
    uploaded = {p: datetime.fromisoformat(r["uploaded_at"]) for p, r in zip(paths, photos)}
    fields = ph.extract_photos(paths, engines, uploaded_at=uploaded, vpic=vpic)
    seconds = time.perf_counter() - t0

    obs = [PhotoObs.from_photo(r["photo"], pf, uploader=r["uploader"]) for r, pf in zip(photos, fields)]
    clusters = cluster_photos(obs)
    decisions = match(clusters, list(inv_obs.values()), vpic=vpic or None)

    # --- Clustering ---
    true_set = {r["photo"]: r["set_id"] for r in photos}
    set_invoice = {r["set_id"]: r["invoice_id"] or None for r in photos}
    set_case = {r["set_id"]: r["case"] for r in photos}
    set_size = Counter(true_set.values())
    groups = [[p.id for p in c.photos] for c in clusters]
    precision, recall = pair_scores(true_set, groups)
    exact = sum(1 for g in groups if len({true_set[p] for p in g}) == 1 and len(g) == set_size[true_set[g[0]]])
    impure = [c for c in clusters if len({true_set[p.id] for p in c.photos}) > 1]
    unassigned = [c for c in clusters if c.status == "unassigned"]
    clusters_per_set = Counter(Counter(true_set[p.id] for p in c.photos).most_common(1)[0][0] for c in clusters)

    # --- Matching ---
    outcome = Counter()
    wrong_links, rows_out = [], []
    set_result: dict[str, list[str]] = defaultdict(list)
    for c, d in zip(clusters, decisions):
        sets_in = Counter(true_set[p.id] for p in c.photos)
        main_set = sets_in.most_common(1)[0][0]
        truth = set_invoice[main_set]
        invoices_in = {set_invoice[s] for s in sets_in}
        outcome[d.status] += 1
        if d.status == "linked":
            ok = invoices_in == {d.invoice_id}
            outcome["linked_right" if ok else "linked_wrong"] += 1
            if not ok:
                wrong_links.append((c.id, d.invoice_id, dict(sets_in), d.reasons))
            verdict = "linked right" if ok else "LINKED WRONG"
        elif d.status == "review":
            in_top = truth is not None and any(x.invoice_id == truth for x in d.candidates)
            outcome["review_true_in_top"] += in_top
            if d.candidates and d.candidates[0].misread:
                outcome["review_misread"] += 1
                outcome["review_misread_right"] += d.candidates[0].invoice_id == truth
            verdict = "review, right invoice in top" if in_top else (
                "review (no invoice)" if truth is None else "review, right invoice not in top")
        else:
            verdict = "unmatched (right: no invoice)" if truth is None else "unmatched (missed)"
            outcome["unmatched_no_invoice" if truth is None else "unmatched_missed"] += 1
        set_result[main_set].append(verdict)
        rows_out.append((c, d, sets_in, truth, verdict))

    n = len(clusters)
    links = outcome["linked"]
    mode = f"vision model {vision.model}" if vision else "no vision model (rules)"
    print(f"\n{len(photos)} photos in {len(set_size)} sets, {len(invoices)} invoices; {mode}\n")
    print("Clustering")
    print(f"  clusters {n} (true sets {len(set_size)}), exactly right {exact}/{len(set_size)}")
    print(f"  pairwise precision {100 * precision:.1f}%, recall {100 * recall:.1f}%")
    print(f"  mixed clusters (photos of two cars) {len(impure)}, unassigned photos {len(unassigned)}, "
          f"sets split in more than one cluster {sum(1 for k in clusters_per_set.values() if k > 1)}")
    print("\nMatching (per cluster)")
    print(f"  auto-linked   {links:3} ({100 * links / n:4.1f}%)   right {outcome['linked_right']}, "
          f"WRONG {outcome['linked_wrong']}")
    print(f"  auto-link precision {100 * outcome['linked_right'] / links:.1f}%" if links else "  no auto-links")
    print(f"  review        {outcome['review']:3} ({100 * outcome['review'] / n:4.1f}%)   right invoice among top "
          f"{outcome['review_true_in_top']}")
    print(f"    of which likely misread invoice VIN/plate {outcome['review_misread']}, "
          f"right invoice on top {outcome['review_misread_right']}")
    print(f"  unmatched     {outcome['unmatched']:3} ({100 * outcome['unmatched'] / n:4.1f}%)   no invoice exists "
          f"{outcome['unmatched_no_invoice']}, missed {outcome['unmatched_missed']}")
    linked_invoices = {d.invoice_id for d in decisions if d.status == "linked"}
    sets_linked = sum(1 for s, inv in set_invoice.items() if inv and inv in linked_invoices
                      and any(v == "linked right" for v in set_result[s]))
    invoiced_sets = sum(1 for inv in set_invoice.values() if inv)
    print(f"  invoiced photo sets linked to their invoice: {sets_linked}/{invoiced_sets}")
    no_photos = [r["invoice_id"] for r in invoices if r["invoice_id"] not in linked_invoices]
    print(f"  invoices left without a linked cluster: {len(no_photos)}")
    print("\nHard cases")
    for s in sorted(set_case, key=lambda s: (set_case[s] == "normal", s)):
        if set_case[s] != "normal":
            print(f"  {s} {set_case[s]:22} invoice {set_invoice[s] or '-':8} -> {', '.join(set_result[s])}")
    no_photo_invoices = sorted({r["invoice_id"] for r in invoices} - {i for i in set_invoice.values() if i})
    for iid in no_photo_invoices:
        print(f"  --  invoice_without_photos invoice {iid:8} -> "
              + ("not linked (right)" if iid not in linked_invoices else "LINKED (wrong)"))
    print(f"\nTime: {seconds:.0f} s extraction (cache misses only run engines)")
    if wrong_links:
        print("\nWRONG AUTO-LINKS:")
        for w in wrong_links:
            print("  ", *w)
    if args.verbose:
        print("\nClusters:")
        for c, d, sets_in, truth, verdict in rows_out:
            print(f"  {c.id} sets {dict(sets_in)} truth {truth or '-'}: {d.status} {d.invoice_id or ''} "
                  f"({d.decided_by or ''}, score {d.score}) -> {verdict}")
            for r in d.reasons:
                print(f"      - {r}")
            for x in d.candidates:
                print(f"      * {x.invoice_id} {x.score:g} " + ", ".join(f"{s.name} {s.points:g}" for s in x.signals)
                      + (f"  conflicts: {'; '.join(x.conflicts)}" if x.conflicts else ""))
        print("\nInvoice files used:", Counter(p.name.split("_")[-1].split(".")[0] if "_" in p.name and
                                                 not p.name.endswith(".pdf") else "pdf" for p in inv_files.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
