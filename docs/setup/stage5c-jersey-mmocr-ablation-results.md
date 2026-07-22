# Stage 5C-C2 — Jersey MMOCR controlled ablation results

- **Status:** `completed_no_exact_signal_in_tested_variants`
- **Pipeline status:** successful
- **Model-family status:** `closed_after_controlled_negative_result`
- **Source stage:** Stage 5C-C2
- **Freeze:**
  `outputs/reid/full_stage4b/jersey_mmocr_ablation_freeze_stage5c_c2`
- **C1 baseline freeze:**
  `outputs/reid/full_stage4b/jersey_mmocr_smoke_baseline_freeze_stage5c_c1`

## Purpose

Measure whether the frozen Stage 5C-C1 DBNet+SAR general scene-text
checkpoint family can produce any exact jersey-digit signal when:

1. the detector is bypassed (direct SAR on the Stage 5A ROI);
2. the ROI is deterministically upscaled 2× / 4× with `INTER_CUBIC`;
3. DBNet+SAR runs on a 4× ROI.

No new checkpoint, dataset, threshold, or fine-tuning was introduced.

## C1 baseline (frozen reference)

| Metric | Value |
|---|---|
| Items | 46 (POS/A/B/C/D/E = 20/10/2/7/2/5) |
| Exact match | 0/20 |
| Positive emission | 0 |
| Negative emission | 0/26 |
| Detector region items | 1 |
| Detector no-region | 45 |
| Network policy | `pass_loopback_only` |
| Preprocessing | `number_search_roi_v1` (no upscale) |

## C2 experiment matrix

Exact four variants, same 46 crops, same DBNet/SAR checkpoints:

1. `direct_sar_roi_1x`
2. `direct_sar_roi_2x_cubic`
3. `direct_sar_roi_4x_cubic`
4. `dbnet_sar_roi_4x_cubic`

Ordering: variant-major, then `pilot_index` ascending → **184**
predictions. Offline loopback-only network policy; Stage 5A
number-search ROI reused in memory; no image export.

## Results (recomputed from freeze artifacts)

| Variant | Exact | Pos. emission | Wrong | No-pred | Neg. emission | Notes |
|---|---:|---:|---:|---:|---:|---|
| C1 DBNet+SAR 1× | 0/20 | 0 | 0 | 20 | 0 | frozen baseline |
| direct SAR 1× | 0/20 | 3 | 3 | 17 | 0 | wrong digits only |
| direct SAR 2× | 0/20 | 2 | 2 | 18 | 0 | |
| direct SAR 4× | 0/20 | 2 | 2 | 18 | 0 | |
| DBNet+SAR 4× | 0/20 | 0 | 0 | 20 | 0 | region items **6**, total regions **52** |

Evidence labels:

- `UPSCALE_IMPROVES_DETECTION`
- `NO_EXACT_SIGNAL_IN_TESTED_VARIANTS`

Raw / rejection highlights:

- Direct SAR positive rejections are dominated by `non_digit_text`
  and occasional `digit_count_exceeds_max` (e.g. `130`, `1997`).
- Accepted direct-SAR digits were wrong and low-confidence
  (roughly 0.17–0.52).
- DBNet 4× SAR region texts were mostly `I` / `-` / short non-digits;
  accepted digit candidate count = 0.

Runtime (C2 run): detector init ≈ 7.2 s; recognizer init ≈ 1.2 s;
inference wall ≈ 228 s; peak RSS ≈ 1.6 GB. Confidences are **not**
calibrated across detector and direct-recognizer pipelines.

## C2a / C2b status

- **C2a** detector-region review package:
  `outputs/reid/full_stage4b/jersey_mmocr_detector_region_review_stage5c_c2a`
  - status: `review_package_generated_unreviewed`
  - 52 regions / 6 overviews / 4 contact sheets
  - `manual_review_completed=false`
- **C2b** Furkan manual region review:
  **`skipped_by_project_decision_not_required_for_c3a`**
  - working CSV exists but all `manual_*` fields remain blank
  - **C2b manual region review was not performed**
  - the 52-region package is present but unreviewed
  - **DBNet localization failure is not manually confirmed**

## Interpretation limits

- Pipeline successful; this is **not** a general OCR accuracy benchmark.
- Tested checkpoint family produced **no exact signal** on the 46-crop
  set.
- 4× upscale increased DBNet region generation but did not create
  exact matches.
- Direct SAR produced only wrong, low-confidence digit emissions.
- Current evidence is judged sufficient to **close** the general
  scene-text DBNet+SAR checkpoint family for further advancement.
- **Jersey OCR as a product direction is not cancelled.**

## Model-family closure and next candidate

- Closed family: ICDAR/scene-text **DBNet + SAR** (MMOCR) used in C1/C2.
- Selected next model family: **SoccerNet-finetuned PARSeq**
  (capability audit: Stage 5C-C3A).
- Next gate: **Stage 5C-C3B** isolated PARSeq CPU environment plan
  (no PARSeq environment or checkpoint was installed/downloaded in
  this documentation gate).

## Freeze contents

Eleven files under
`outputs/reid/full_stage4b/jersey_mmocr_ablation_freeze_stage5c_c2`:

- nine byte-identical C2 artifacts
- `ablation_freeze_summary.json`
- `ablation_freeze_manifest.json` (self-hash omitted)

Tracked application files for C2 (committed separately):

- `src/football_analytics/reid/jersey_mmocr.py`
- `scripts/run_reid_jersey_mmocr_ablation.py`
- `configs/reid/jersey_mmocr_ablation_stage5c_c2.yaml`
- `tests/test_reid_jersey_mmocr_ablation.py`
