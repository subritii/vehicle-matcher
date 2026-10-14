# Matching on dev (synthetic_hard)

- Date: 2026-10-13. First version in `8e80b46`; the misread rule is in the commit that updates this file.
- Data: `python tools/make_fixtures.py --difficulty hard --matching-only`. There are 148 photos in 37 photo sets
  for the 36 hard dev invoices (layouts 1-3); holdout was not touched. Exteriors are Wikimedia Commons photos.
  VIN, plate and odometer shots are generated. 43 of 148 photos have no EXIF, so their time comes from the
  upload time at half weight.
- Invoices: each one is read as a single file, cycling through PDF, scan and phone photo (12 of each).
- Command: `python tools/match_accuracy.py --no-vision` and `--model qwen3.5:4b-q4_K_M`.
- Weights: `app/match/config.py`, the starting weights from the design rules, not tuned. Auto-link needs a VIN or plate
  trusted on both sides, or a score of at least 80 with a lead of at least 25 and no conflict. Review covers
  40 to 79, or a close top two. Anything lower is unmatched. A likely misread invoice VIN or plate always
  goes to review, with that invoice on top (see below).

## Results

| | Rules only | Rules + vision model |
|---|---|---|
| Clusters (true sets 37) | 41 | 39 |
| Clusters exactly right | 33/37 | 35/37 |
| Pairwise precision / recall | 100.0% / 94.3% | 100.0% / 97.4% |
| Mixed clusters (two cars in one) | 0 | 0 |
| **Wrong auto-links** (target 0) | **0** | **0** |
| **Auto-link precision** (target 100%) | **100% (14/14)** | **100% (20/20)** |
| Clusters auto-linked | 14/41 (34.1%) | 20/39 (51.3%) |
| Clusters sent to review | 11/41 (26.8%), right invoice in top 3: 7 | 9/39 (23.1%), right invoice in top 3: 7 |
| Clusters unmatched | 16/41 (39.0%): 2 correct (no invoice), 14 missed | 10/39 (25.6%): 2 correct, 8 missed |
| Invoiced photo sets linked to their invoice | 14/35 | 20/35 |

The first run filled the engine cache, taking 780 s rules-only and 3,657 s with the model. A rerun from
`data/extract_cache/` takes 16 s and gives the same numbers.

## Hard cases

| Case | Rules only | Rules + vision model |
|---|---|---|
| Two look-alikes (white Ford Escape, same shop, uploader and day, 12-20 min apart; one has no invoice) | original: review, right invoice in top; twin: unmatched (correct) | original: linked right on VIN; twin: unmatched (correct) |
| Photo set with no VIN or plate shot (v01, v12) | both unmatched (best score below 40) | both unmatched (scores 18 and 31) |
| No EXIF, two sets uploaded together (v08, v14) | v14: linked right, its odometer photo goes to review as ambiguous; v08: unmatched, plus one photo to review | v14: linked right on VIN; v08: review with the right invoice on top (plate cut short), plus one photo to review |
| Photo set with no invoice (v37) | unmatched (correct) | unmatched (correct) |
| Invoice with no photos (inv_002) | not linked (correct) | not linked (correct) |

When a hard case cannot be decided safely, it goes to review or stays unmatched; none of them is linked
wrong. The look-alike twin never takes the original's invoice, because its trusted VIN and plate veto it.

## Likely misread invoice VIN or plate (review, never auto-link)

The rule applies when a cluster has a trusted photo VIN and an unlinked invoice has an untrusted VIN at most
3 edits away. It also applies when a cluster has a trusted photo plate and the invoice plate is cut short
(at least 4 characters) or 1 edit away. In both cases make/model (vPIC) or color must agree. The cluster then
goes to review with that invoice on top and the reason "invoice VIN likely misread" or "invoice plate likely
misread". A cut-short plate no longer vetoes the invoice. A cluster with such a candidate never auto-links
on score. Settings are in `app/match/config.py` (`MISREAD_*`).

| | Rules only, before | Rules only, after | + model, before | + model, after |
|---|---|---|---|---|
| **Wrong auto-links** | **0** | **0** | **0** | **0** |
| Auto-linked | 14 (34.1%) | 14 (34.1%) | 20 (51.3%) | 20 (51.3%) |
| Review | 11 (26.8%) | 11 (26.8%) | 6 (15.4%) | 9 (23.1%) |
| ... right invoice in top 3 | 7 | 7 | 4 | 7 |
| Unmatched, missed | 14 | 14 | 11 | 8 |

With the model, 3 missed clusters move from unmatched to review, each with the right invoice on top:

- inv_048: VIN `1PMCU0G6110675311` vs photo `1FMCU0G61LU675311` (3 edits).
- inv_003: VIN `1C6SRFFTJFM758261` vs photo `1C6SRFFT3PN758261` (3 edits).
- inv_013: plate `EN90` vs photo `EN90604` (cut short; this used to be a veto).

Rules-only moves none. inv_048 and inv_003 are within 3 edits there too, but nothing else agrees.
Without the model the photos have no color, and the invoice make/model is either missing or disagrees
with vPIC. The agreement condition is what keeps the rule from matching a VIN by edit distance alone.

Two misses with the model are out of the rule's reach. inv_033's invoice VIN is 6 edits off
(`M9A5B10011P404985`), and inv_018's "VIN" is the invoice number read as a VIN.

## Why sets are missed

The misses come from invoice extraction, not from clustering or scoring. For every missed invoiced set, the
photo side reads the right VIN or plate. The right invoice still loses its identifier for one of these reasons:

- The VIN on a scan or phone photo is garbled and fails its check digit. Examples are
  `1PMCU0G6110675311` for `1FMCU0G61LU675311`, `M9A5B10011P404985` for `WBA5R1C01LF404985`, and once an
  invoice number read as a VIN.
- The invoice plate is missing or cut short. inv_013 reads `EN90` for `EN90604`, which vetoes the right
  invoice.
- The invoice has no plate and its VIN was not extracted (inv_053), so the score rests on make/model, color, date and
  odometer. Those add up to at most 43, below the 80 needed to auto-link.

A garbled VIN counts as an identifier conflict, so it blocks auto-linking but subtracts no points. This is
the safe choice: no score-based rule could turn these cases into links without the risk of wrong links. The fix
belongs in invoice VIN and plate extraction from scans and photos, and those clusters then go to review. Tuning the
weights on these dev results would only overfit.

Sets with no VIN or plate shot (v01, v12) can score at most 43 even when everything else agrees, so they
never auto-link. That is intended.

## Caveats

- The photos are generated: rendered VIN plates, stickers, odometers and plates, plus real Commons
  exteriors. Real photos will be harder to read.
- One no-EXIF batch pairs two invoices from the same vendor about a month apart (inv_013 and inv_023), the
  nearest dates available on dev. A real batch would usually be closer.
- The "same uploader/location +5" signal is not implemented, because there is no vendor-to-location table.
  Uploader and device are used only for clustering.
- The auto-link lead is measured against every invoice that is not vetoed, including ones already linked.
  This is stricter than the design rules ask, on purpose: a cluster that looks like a vehicle already linked goes to review.
- Ambiguous photos are odometer shots taken minutes apart by the same uploader. They go to review as
  single-photo clusters, so they count as split sets in the clustering numbers.
- With 37 sets, one set moves a percentage by about 3 points.
