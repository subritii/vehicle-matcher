# Baseline: rules only (no vision model), hard dev split

- Date: 2026-10-12
- Commit: 87c9709 (OCR worker process with per-image timeout)
- Command: `OCR_TIMEOUT=120 python tools/accuracy.py --difficulty hard -v`
- Settings: `OCR_ENGINE=paddle` (PP-OCRv6 medium det/rec), one OCR worker process,
  `OCR_TIMEOUT=120` s per image, `VIN_MARKET=north_america`, vPIC on, `OLLAMA_MODEL` empty
- Data: `tests/fixtures/synthetic_hard/dev/` (layouts 1-3), 36 invoices x 3 renderings
- OCR timeouts: 0. OCR failures: 0.

| field | pdf | scan | photo |
|---|---|---|---|
| vin | 32/32 100.0% | 18/32 56.2% (+1F) | 20/32 62.5% (+1F) |
| plate | 25/25 100.0% | 2/25 8.0% (+2F) | 6/25 24.0% |
| plate_state | 25/25 100.0% | 3/25 12.0% | 6/25 24.0% |
| year | 36/36 100.0% | 20/36 55.6% | 24/36 66.7% |
| make | 36/36 100.0% | 20/36 55.6% | 23/36 63.9% |
| model | 36/36 100.0% | 16/36 44.4% | 18/36 50.0% |
| color | 29/31 93.5% | 10/31 32.3% | 14/31 45.2% |
| mileage | 27/36 75.0% | 5/36 13.9% | 11/36 30.6% |
| invoice_date | 36/36 100.0% | 17/36 47.2% | 28/36 77.8% |
| vendor | 36/36 100.0% | 29/36 80.6% | 33/36 91.7% |
| invoice_no | 36/36 100.0% | 24/36 66.7% | 23/36 63.9% |
| total | 36/36 100.0% | 25/36 69.4% | 28/36 77.8% |
| **all fields** | **390/401 97.3%** | **189/401 47.1%** | **234/401 58.4%** |
| needs review | 0/36 | 21/36 | 17/36 |
| VIN trusted | 32/36 | 13/36 | 18/36 |
| wrong + trusted VIN | 0 | 0 | 0 |
| sec/invoice | 15.76 | 11.14 | 12.93 |

`+NF`: answer key blank but a value was extracted (N invoices).
Timings include OCR unless it was served from `data/ocr_cache/`; cache state before this run was not recorded.
