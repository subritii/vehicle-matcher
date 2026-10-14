# Photo extraction: rules-only vs rules + vision model (dev)

- Date: 2026-10-13
- Code: parent commit 7554f78 plus the photo-side changes in this commit
- Command: `python tools/photo_accuracy.py --no-vision -v` and
  `python tools/photo_accuracy.py --model qwen3.5:4b-q4_K_M -v`
- Settings: OCR_ENGINE=paddle (one worker, OCR_TIMEOUT=120), fast-alpr 0.4.0
  (yolo-v9-t-384 detector, cct-xs-v2-global OCR), zxing-cpp 3.1.1, vPIC on,
  Ollama `qwen3.5:4b-q4_K_M`. OCR, fast-alpr and the model never run at the same time:
  EXIF + barcode, then the model, then OCR, then fast-alpr. Each engine is unloaded before the next starts.
- Machine: M1, 8 GB

## Photos (204, dev only)

| Set | Photos | Source |
|---|---|---|
| Synthetic (`make_fixtures.py --difficulty hard`) | 150 | 36 each of VIN plates, door-jamb stickers (Code 39 barcode), odometers and plate close-ups, for the 36 hard dev invoices' vehicles, plus 6 invoice photos labeled "other". 75% have EXIF time, GPS and device. |
| Wikimedia Commons (`fetch_exteriors.py`) | 54 | Exteriors of the dev vehicles' make, model and color. Labels come from Commons color categories. Licenses and authors are in `tests/fixtures/exteriors/exteriors.csv`. |

There are no interior photos and no real photos yet (`tests/fixtures/private/photos/` is empty).

## Per photo type

| True shot type | Photos | Shot type (rules) | Shot type (+model) | Field | Rules only | + model |
|---|---|---|---|---|---|---|
| exterior | 54 | 19 (35%) | **54 (100%)** | color | 0/54 | **51/54** |
| | | | | body_style | 0/54 | **52/54** |
| vin_plate | 36 | 34 (94%) | 34 (94%) | vin | 34/36 | 34/36 |
| door_jamb | 36 | 36 (100%) | 36 (100%) | vin | 36/36 | 36/36 |
| odometer | 36 | 36 (100%) | 36 (100%) | odometer | 36/36 | 36/36 |
| plate | 36 | 36 (100%) | 36 (100%) | plate | 34/36 | 34/36 |
| | | | | plate_state | 0/36 | **34/36** |
| other | 6 | 0 (0%) | **6 (100%)** | | | |
| **all** | 204 | 161 (79%) | **202 (99%)** | | | |

EXIF on the 150 synthetic photos, the same in both runs: capture time 144/144 (104 from EXIF, 40 from the upload-time fallback, which gets a lower weight), GPS 104/104, device 104/104.

## Trust (target: 0 wrong + trusted)

| | Rules only | + model |
|---|---|---|
| VIN trusted (may auto-link) | 19 (barcode) | 66 (19 barcode + 47 OCR confirmed by the model) |
| VIN wrong + trusted | **0** | **0** |
| Plate trusted | 0 | 34 |
| Plate wrong + trusted | **0** | **0** |
| Photos needing review | 111 | 30 |
| Time | 5.9 s/photo | 25.4 s/photo |

Where values came from (right / wrong), with the model: VIN from barcode 19/0, VIN from OCR 51/0, plate from ALPR 34/2, plate state from the model 34/0, odometer from OCR 36/0, color from the model 51/3, body style from the model 52/2.

## Trust rules
- **VIN**: same rules as invoices. A barcode with a valid check digit is trusted. An OCR VIN is trusted only when the model reads the same VIN and there are no issues (check digit, repair, unknown WMI, several VINs). A model-only VIN is kept only if it passes the check digit and matches an OCR candidate or decodes cleanly in vPIC, and even then it is never trusted.
- **Plate** (plates have no check digit): trusted only when all of these hold:
  - read by fast-alpr on a **plate close-up**
  - the model reads the same plate
  - the plate has **no O, I or Q**

  Everything else goes to review.

## What changed after the first run
The first run with the model gave **11 wrong + trusted plates**, and VIN accuracy dropped to 37/72. These were fixed (tests first) and the model run was repeated:
1. **The two readers agreeing is not enough for plates.** fast-alpr and the model both read `OFYD327` for `0FYD327`.
   - Fix: plates with O, I or Q are never trusted.
2. **On exteriors**, both readers agreed on badges and frames: `2009`, `2019`, `3689`, `MODEL3`.
   - Fix: plates from exteriors are never trusted. They still feed clustering and review.
   - The Commons key has no plate labels, so some of the other "wrong" exterior plates may have been real plates. They count as wrong either way.
3. **The model called 35/36 synthetic VIN plates "other"**, so OCR never looked at them.
   - Fix: OCR now also runs on photos the model calls "other" or "interior". If the text has ≤ 12 lines and contains a VIN, the rules set the shot type. Invoice photos (21–33 OCR lines) stay "other".
   - Cost: one extra OCR call per such photo.

## Remaining errors (with the model)
- **inv_002 and inv_052 VIN plates**: OCR reads no complete VIN, and the model reads none either. Both runs miss these two.
- **inv_008 and inv_021 plates**: fast-alpr reads O for 0. Not trusted (O rule); the plate goes to review.
- **Plate state**: 2 missing (inv_008, inv_031). The state is only kept when the plates agree.
- **Color**: 3 wrong. One is gray read as silver, one silver as white, and one gray Tesla as black. Commons color categories are noisy, and gray/silver is a judgment call.
- **Body style**: 2 wrong, both Subaru Outback read as SUV instead of wagon. Debatable.

## Caveats
- The synthetic VIN plates are flat renderings. The model doesn't recognize them as dashboard VIN plates, so the shot-type gain on VIN plates is untested on real photos.
- The generator's door-jamb stickers were made more readable while building this:
  - a larger Code 39 barcode
  - less blur

  At first, zxing read 0/36 barcodes, so these changes were needed. It now reads 19/36. Barcode results on synthetic stickers are therefore optimistic.
- Commons labels: one "blue Camry" photo looks dark. No free silver RAV4 photo was found.
- Rules only, without the model: exteriors are "other" 28 times and "odometer" 7 times, and invoice photos are called VIN plates. Color, body style and plate state need the model.
- The model is slow: ~20 s per photo on an M1 8 GB. For a real batch, the model could skip photos whose shot type the rules already know (barcode, an odometer read, a plate close-up).
