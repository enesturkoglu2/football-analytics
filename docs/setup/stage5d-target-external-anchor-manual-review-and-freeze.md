# Stage 5D-B1E-E — External Anchor Manual Review and Freeze

## Purpose

Freeze human decisions for the 15 external anchor crop candidates and
promote exactly seven `target_anchor_yes` crops to frozen anchors.

`target_anchor_no` means **not selected for anchor membership**
(`not_selected_redundant_valid_target_crop`). It is **not** an identity
negative and must not enter distractor/evaluation-negative sets.

## Approved anchors (exact)

1. `target_001_ext_anchor_001`
2. `target_001_ext_anchor_003`
3. `target_001_ext_anchor_004`
4. `target_001_ext_anchor_006`
5. `target_001_ext_anchor_008`
6. `target_001_ext_anchor_011`
7. `target_001_ext_anchor_014`

## Status

`COMPLETED_STAGE5D_B1E_E_TARGET_001_EXTERNAL_ANCHORS_FROZEN`

Exact next gate:
`STAGE5D-B1E-F_TARGET_001_FROZEN_ANCHOR_OSNET_EMBEDDING_AND_GALLERY_BUILD`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_anchor_manual_freeze.py \
  --config configs/reid/external_anchor_manual_freeze_stage5d_target_001.yaml
```
