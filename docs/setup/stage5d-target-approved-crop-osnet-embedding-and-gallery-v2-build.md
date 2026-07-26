# Stage 5D-F3G — Approved Crop OSNet Embedding and Gallery-v2 Build

## Purpose

- Reuse gallery-v1 (7) immutably
- Embed 6 approved new target crops + 23 approved hard-negative crops
- Build target gallery-v2 (13) and same-team distractor gallery-v1 (23)
- Diagnostic centroid/medoid/pairwise/cross only

No sample evaluation, threshold, identity assignment, or gallery-v1 mutation.

## Status / readiness

- `COMPLETED_STAGE5D_F3G_TARGET_001_GALLERY_V2_AND_DISTRACTOR_GALLERY_BUILT`
- `TARGET_001_GALLERY_V2_AND_DISTRACTOR_GALLERY_READY_FOR_SCORING_DESIGN`

Exact next gate:
`STAGE5D-F3H_TARGET_001_TARGET_DISTRACTOR_SCORING_CONTRACT_AND_NEW_HOLDOUT_DESIGN`

## Runtime

```bash
conda run -n sn-reid-cpu python scripts/run_reid_target_gallery_v2_and_distractor_gallery_build.py \
  --config configs/reid/target_gallery_v2_and_distractor_gallery_stage5d_target_001.yaml
```
