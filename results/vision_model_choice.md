# Choosing the local vision model

- Date: 2026-10-12
- Machine: Apple M1, 8 GB RAM, Ollama 0.34.4. OCR worker stopped during the test.
- Requirement: about 4 GB or less, vision input, can return JSON fields.

## Candidates in the Ollama library (checked 2026-10-12)

| model | size | notes |
|---|---|---|
| `qwen3.5:4b-q4_K_M` | 3.3 GB | general vision-language model, JSON output |
| `minicpm-v4.6` (1b) | 1.6 GB | general vision-language model, smallest |
| `qwen3.5:2b-q4_K_M` | 1.9 GB | smaller sibling of the 4b |
| `glm-ocr` | 2.2 GB | OCR-only model: returns page text, not fields (we already have PaddleOCR) |
| `gemma4:e2b` / `e4b` | 4.3 GB+ | over budget |

`qwen3.5:4b` and `minicpm-v4.6` were tested as the two best general models that fit.

## Test: 5 hard dev images, all 12 fields

Images: inv_001 scan, inv_002 photo, inv_053 scan, inv_057 photo, inv_058 scan.
Each model answer was parsed with the same parsers the pipeline uses. Images were
downscaled to 1280 px on the long side. Temperature 0, `think: false`.

| | qwen3.5:4b-q4_K_M | minicpm-v4.6 |
|---|---|---|
| fields right | **42/55 (76.4%)** | 24/55 (43.6%) |
| VINs right (check digit valid and correct) | 1/5 | 0/5 |
| wrong VINs that passed the check digit | 0 | 0 |
| first image (load + read) | 33.5 s | 11.7 s |
| later images (model loaded) | 23.3 s | 7.1 s |
| loaded size (`ollama ps`) | 3.2 GB, 100% GPU | 0.7 GB, 100% GPU |
| swap used, before -> during | 2.4 GB -> 5.6 GB | 4.3 GB -> 3.6 GB |

Notes:
- Both models misread most VINs on the hard images, but every misread failed the check
  digit, so the pipeline drops it. That is why the VIN gates matter more than the model.
- qwen also read dates in the wrong year (`02/04/25`, `2020-08-23`) and one wrong total
  (`276.20` vs `278.20`). Those are the risk for fields with no check digit.
- minicpm left many fields empty or put the wrong field in (mileage taken from a
  money amount, plate taken from the VIN line).
- The process RSS doesn't show the model's memory on Apple Silicon (Metal unified memory),
  so `ollama ps` size and system swap are reported instead. The 8 GB machine swaps under
  qwen. That is why OCR and the model never run together.

**Choice: `qwen3.5:4b-q4_K_M`**, nearly twice as accurate, at about 3x the time per image.
