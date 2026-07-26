# Stage 5D-F3E — External Refinement Crop Quality and Hard-Negative Review Package

## Purpose

Prepare human crop-review artifacts from F3D-frozen sources only:

1. Immutable reference sheet for 9 target sources (7 gallery-v1 + 2
   approved expansion crops).
2. EXT_161 target crop candidates (1–4, quality + temporal diversity).
3. Exact one hard-negative crop candidate per 35 same-team distractor
   sources (no silent drops; quality-exception fallback allowed).

No crop approvals, embeddings, gallery-v2, hard-negative membership,
OCR/similarity, or sample access.

## Limits

- EXT_161 candidates: 1–4
- distractor candidates: exact 35 (1 per source)
- contact sheets min width: 3600 px
- HN sheets: 12 / 12 / 11
- approvals / embeddings / gallery mutation: 0

## Status

`COMPLETED_STAGE5D_F3E_TARGET_001_EXTERNAL_REFINEMENT_CROP_REVIEW_READY`

Exact next gate:
`STAGE5D-F3F_TARGET_001_EXTERNAL_REFINEMENT_CROP_MANUAL_REVIEW_AND_FREEZE`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_refinement_crop_review_package.py \
  --config configs/reid/external_refinement_crop_review_stage5d_target_001.yaml
```
