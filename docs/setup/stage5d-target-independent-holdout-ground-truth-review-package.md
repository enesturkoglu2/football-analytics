# Stage 5D-F3K — Independent holdout v2 ground-truth review package

## Purpose

Similarity-blind / label-blind human ground-truth review package for the frozen
F3J holdout v2 segment universe. No gallery, OSNet, similarity, scoring, ranking,
metrics, threshold, or identity assignment. Manual decision fields remain blank.

## Status

- Final: `COMPLETED_STAGE5D_F3K_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_REVIEW_PACKAGE_READY`
- Readiness: `TARGET_001_INDEPENDENT_HOLDOUT_V2_GROUND_TRUTH_READY_FOR_HUMAN_REVIEW`
- Next: `STAGE5D-F3L_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_MANUAL_REVIEW_AND_FREEZE`

## Frozen inputs

- F3J universe: 243 segments · 141 eligible · 102 ineligible
- Holdout: `data/test_clips/target_001_independent_holdout_v2.mp4`
- Crop padding: 0.05 (canonical Stage 5D external crop contract)

## Outputs

- 141 review items `H2_GT_REVIEW_000001`…`000141`
- 141 representative crop PNGs
- 12 contact sheets (11×12 + 9)
- 3 diagnostic review videos (47/47/47)
- Empty GT CSV template (141 rows, all manual fields blank)
- Separate ineligible inventory (102; not automatic negatives)

## Run

```bash
conda run -n football-cv python scripts/run_reid_independent_holdout_v2_ground_truth_review_package.py \
  --config configs/reid/independent_holdout_v2_ground_truth_review_stage5d_target_001.yaml
```

## Tests

```bash
conda run -n football-cv python -m unittest discover -s tests \
  -p 'test_reid_independent_holdout_v2_ground_truth_review_package.py' -v
```
