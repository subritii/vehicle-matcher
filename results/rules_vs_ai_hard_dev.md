# Rules-only vs rules + vision model, hard dev split

- Date: 2026-10-13
- Code: the commit that adds this version (parent 37fb3ad)
- Command: `OCR_TIMEOUT=120 python tools/accuracy.py --difficulty hard --vision --model qwen3.5:4b-q4_K_M -v`
- Settings: `OCR_ENGINE=paddle`, one OCR worker, `OCR_TIMEOUT=120`, vPIC on,
  vision model `qwen3.5:4b-q4_K_M` (see `vision_model_choice.md`), images 1280 px long side,
  temperature 0, `VISION_FILL_BELOW=0.5`, model values get confidence 0.5
- Data: `tests/fixtures/synthetic_hard/dev/` (layouts 1-3), 36 invoices x 3 renderings
- Order: rules on all 108 files, then the OCR worker is stopped, then one model session
  (unloaded at the end). OCR and the model never ran together. OCR timeouts: 0.
- Rules-only numbers match `baseline_rules_only.md`, except sec/invoice: OCR came from the cache this time.

Safety rules in this version:
- Date, invoice no., total, mileage (no check digit): if the OCR value and the model
  value disagree, the OCR value stays and the invoice goes to review with both values.
  The model may still fill a field the OCR left blank.
- Plausibility: invoice date not in the future and not before 2000, mileage under
  1,000,000, total above zero. An implausible model value is dropped; an implausible
  rules value goes to review.
- A model plate is kept only if the OCR text has a plate label (Plate, Lic, License,
  Reg, Tag, matched at a word start so OCR-garbled labels like "LICA" count) with a
  similar string on that line or within the next two lines. A model-only plate has
  `plate_trusted = False`, so matching must never auto-link on it. Digit-only plates are allowed.
- VIN: unchanged (check digit + OCR match or vPIC with printed year/make; a model-only VIN is never trusted).

```
Rules-only vs rules + qwen3.5:4b-q4_K_M (cells: rules -> rules+AI)

field                              pdf                    scan                   photo
--------------------------------------------------------------------------------------
vin                   100.0% -> 100.0%  56.2% -> 56.2% (1F/1F)  62.5% -> 62.5% (1F/1F)
plate                 100.0% -> 100.0%    8.0% -> 8.0% (2F/2F)          24.0% -> 40.0%
plate_state           100.0% -> 100.0%          12.0% -> 12.0%          24.0% -> 44.0%
year                  100.0% -> 100.0%          55.6% -> 91.7%          66.7% -> 80.6%
make                  100.0% -> 100.0%          55.6% -> 88.9%          63.9% -> 86.1%
model                 100.0% -> 100.0%          44.4% -> 86.1%          50.0% -> 80.6%
color                  93.5% -> 100.0%          32.3% -> 87.1%          45.2% -> 90.3%
mileage                 75.0% -> 80.6%          13.9% -> 88.9%          30.6% -> 69.4%
invoice_date          100.0% -> 100.0%          47.2% -> 66.7%          77.8% -> 80.6%
vendor                100.0% -> 100.0%          80.6% -> 80.6%          91.7% -> 91.7%
invoice_no            100.0% -> 100.0%          66.7% -> 69.4%          63.9% -> 77.8%
total                 100.0% -> 100.0%          69.4% -> 94.4%          77.8% -> 91.7%
--------------------------------------------------------------------------------------
all fields              97.3% -> 98.3%          47.1% -> 72.3%          58.4% -> 76.3%
needs review                    0 -> 0                21 -> 22                17 -> 15
VIN trusted                   32 -> 32                13 -> 15                18 -> 19
wrong+trusted                   0 -> 0                  0 -> 0                  0 -> 0
sec/invoice                0.1 -> 13.0             0.1 -> 23.2             0.1 -> 22.0

Fields taken from the vision model (right / wrong):
  pdf    4 right, 0 wrong
  scan   101 right, 27 wrong
  photo  72 right, 25 wrong

OCR vs model disagreements left for review (date, invoice no., total, mileage):
  pdf 0, scan 6, photo 5
Model-only plates (kept, never auto-linked), right / wrong:
  pdf 0/0, scan 0/0, photo 4/1
```

Cells: rules-only -> rules+AI. `(aF/bF)`: invoices where the key is blank but a value was
extracted. sec/invoice for rules+AI is mostly model time: ~23 s for each image the model
reads on an 8 GB M1.

## Full rules+AI table

```
field                        pdf              scan             photo
--------------------------------------------------------------------
vin                 32/32 100.0%  18/32  56.2% +1F  20/32  62.5% +1F
plate               25/25 100.0%   2/25   8.0% +2F      10/25  40.0%
plate_state         25/25 100.0%       3/25  12.0%      11/25  44.0%
year                36/36 100.0%      33/36  91.7%      29/36  80.6%
make                36/36 100.0%      32/36  88.9%      31/36  86.1%
model               36/36 100.0%      31/36  86.1%      29/36  80.6%
color               31/31 100.0%      27/31  87.1%      28/31  90.3%
mileage             29/36  80.6%      32/36  88.9%      25/36  69.4%
invoice_date        36/36 100.0%      24/36  66.7%      29/36  80.6%
vendor              36/36 100.0%      29/36  80.6%      33/36  91.7%
invoice_no          36/36 100.0%      25/36  69.4%      28/36  77.8%
total               36/36 100.0%      34/36  94.4%      33/36  91.7%
--------------------------------------------------------------------
all fields        394/401  98.3%    290/401  72.3%    306/401  76.3%
needs review                0/36             22/36             15/36
VIN trusted                32/36             15/36             19/36
wrong+trusted                  0                 0                 0
sec/invoice                13.04             23.23             22.01
```

## Compared with the first version (37fb3ad, no conflict/plausibility/plate rules)

| | first version | this version | rules-only |
|---|---|---|---|
| invoice_date, photo | 69.4% | **80.6%** | 77.8% |
| invoice_no, scan | 63.9% | **69.4%** | 66.7% |
| all fields, scan | 75.1% | 72.3% | 47.1% |
| all fields, photo | 77.8% | 76.3% | 58.4% |
| plate, scan / photo | 24.0% / 44.0% | 8.0% / 40.0% | 8.0% / 24.0% |
| model values right / wrong | 203 / 86 | 177 / 52 | - |
| needs review, scan / photo | 15 / 10 | 22 / 15 | 21 / 17 |
| wrong + trusted VINs | 0 | 0 | 0 |
| false plates from the model (key blank) | 2 | 0 | - |

## Takeaways

- The two fields that got worse in the first version recovered and are now above
  rules-only: invoice_date on photos 77.8% -> 80.6%, invoice_no on scans 66.7% -> 69.4%.
- **Wrong-but-trusted VINs: 0** on every rendering.
- 11 disagreements (6 scan, 5 photo) now go to review instead of being decided by the
  model, so needs review rises compared with the first version. That is the intended trade.
- Model-only plates: 4 right and 1 wrong (inv_043 photo, `60317` instead of `YML7883`),
  all with `plate_trusted = False`. The two invented plates from the first version are gone.
- Scan plates gain nothing. On most scans the plate label is too garbled for the gate
  ("SICE" for "LIC#"), or the OCR misread the plate with high confidence, so the model
  is never asked (inv_001 `71390` for `PBV1390`).
- The model gave 229 values: 177 right and 52 wrong (77%), up from 70%.
