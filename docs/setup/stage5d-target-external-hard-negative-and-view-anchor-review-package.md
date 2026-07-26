# Stage 5D-F3C — External Hard-Negative and View-Anchor Review Package

## Purpose

Prepare a human review package for target_001 external refinement:

1. Additional target-view crop candidates from frozen occurrences
   `EXT_004` / `EXT_183` / `EXT_198` (not anchors until freeze).
2. Neutral review package for 135 unreviewed review-eligible EXT tracks
   (same-team distractor / additional target / other player).

This gate does **not** approve anchors, hard-negatives, embeddings,
thresholds, identities, or gallery-v2.

## Hard rules

- External enrollment video only
- Existing B1E-B detection/tracking lineage only
- Existing B1E-D observation quality diagnostics only
- Sample.mp4 / sample crops / sample embeddings / F3 item scores forbidden
- No YOLO / ByteTrack / OSNet / OCR / similarity inference
- No gallery-v1 mutation
- Manual fields blank until F3D human review

## Limits

- target-view candidates ≤ 4 per occurrence, ≤ 12 total
- prior 15 B1E-D candidate frames excluded
- frozen 7-anchor near-duplicates suppressed (dHash)
- occurrence sheets: 11×12 + 3
- candidate crop copies ≤ 12
- source video / representative crop copies = 0

## Status

`COMPLETED_STAGE5D_F3C_TARGET_001_EXTERNAL_REFINEMENT_REVIEW_READY`

Exact next gate:
`STAGE5D-F3D_TARGET_001_EXTERNAL_REFINEMENT_MANUAL_REVIEW_AND_FREEZE`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_refinement_review_package.py \
  --config configs/reid/external_refinement_review_stage5d_target_001.yaml
```
