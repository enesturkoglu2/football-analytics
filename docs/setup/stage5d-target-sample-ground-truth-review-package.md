# Stage 5D-F2 — Sample Ground-Truth Review Package

## Purpose

Build a **similarity-blind / label-blind** human review package for the
150 scoreable sample.mp4 segmented ReID units against frozen
`target_001`, without computing gallery↔sample scores or filling
ground-truth decisions.

## Expected counts

| Item | Value |
|---|---|
| scoreable evaluation items | 150 |
| unscoreable no-embedding items | 141 |
| contact sheets | 13 (12×12 + 6) |
| manual ground-truth decisions | 0 |
| similarity / ranking rows | 0 |
| gallery members | 7 (unchanged) |

## Visual package

`review_packages/target_001_sample_ground_truth_review/`

- `sample_ground_truth_sheet_01.png` … `_13.png`
- each item: `SAMPLE_EVAL_###`, large crop, START/REP/END context
- no raw/global ID, jersey suggestion, similarity, rank, OCR, or Stage 5C label

## Blank template

`templates/target_001_sample_ground_truth_review_template.csv` — 150 blank rows.

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_sample_ground_truth_review_package.py \
  --config configs/reid/sample_ground_truth_review_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F2_TARGET_001_SAMPLE_GROUND_TRUTH_REVIEW_READY`

Exact next gate:
`STAGE5D-F2A_TARGET_001_SAMPLE_GROUND_TRUTH_MANUAL_REVIEW_AND_FREEZE`
