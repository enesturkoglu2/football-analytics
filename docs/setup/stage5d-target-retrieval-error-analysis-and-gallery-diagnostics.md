# Stage 5D-F3A — Retrieval Error Analysis and Gallery Diagnostics

## Purpose

Analyze F3 independent-retrieval false positives and low-ranked
positives using frozen scores/ranks/GT only. Produce diagnostic
contact sheets and videos. Hypotheses are diagnostic-only; no
gallery mutation, threshold, or identity assignment.

## Forbidden

- Score recompute / formula change
- Gallery add/remove/reweight
- Sample crops as enrollment anchors
- Same-sample gallery-v2 success claim

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_target_retrieval_error_analysis.py \
  --config configs/reid/target_retrieval_error_analysis_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F3A_TARGET_001_RETRIEVAL_ERROR_ANALYSIS_READY`

Exact next gate:
`STAGE5D-F3B_TARGET_001_RETRIEVAL_ERROR_MANUAL_REVIEW_AND_REFINEMENT_DESIGN`
