# Stage 5D-B1E-D — External Tracklet Quality and Anchor Review Package

## Purpose

From frozen positive occurrences `EXT_004`, `EXT_183`, and `EXT_198`,
derive a bounded, temporally diverse set of full-body crop candidates
using existing ByteTrack bbox lineage only.

This gate does **not** approve anchors, compute OSNet embeddings, run
OCR/similarity, or create gallery membership.

## Limits

- max 6 candidates per occurrence
- max 18 total
- unreviewed EXT codes: source read count = 0

## Status

`COMPLETED_STAGE5D_B1E_D_TARGET_001_EXTERNAL_ANCHOR_REVIEW_READY`

Exact next gate:
`STAGE5D-B1E-E_TARGET_001_EXTERNAL_ANCHOR_MANUAL_REVIEW_AND_FREEZE`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_anchor_review_package.py \
  --config configs/reid/external_anchor_review_stage5d_target_001.yaml
```
