# Stage 5D-B1E-C — External Positive Occurrence Freeze

## Purpose

Freeze three human-verified external ByteTrack occurrences for
`target_001` without creating crops, embeddings, gallery members, or
identity assignments.

Selected codes (exact):

- `EXT_004`
- `EXT_183`
- `EXT_198`

Remaining 245 codes stay `unreviewed` and must not be treated as
negatives, distractors, or gallery candidates.

## Review scope

`positive_occurrence_selection_only`

## Status

`COMPLETED_STAGE5D_B1E_C_TARGET_001_EXTERNAL_OCCURRENCES_FROZEN`

Exact next gate:
`STAGE5D-B1E-D_TARGET_001_EXTERNAL_TRACKLET_QUALITY_AND_ANCHOR_REVIEW_PACKAGE`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_positive_occurrence_freeze.py \
  --config configs/reid/external_positive_occurrence_freeze_stage5d_target_001.yaml
```
