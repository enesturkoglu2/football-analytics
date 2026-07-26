# Stage 5D-F3H — Target–Distractor Scoring Contract and New Holdout Design

## Purpose

Preregister target-versus-distractor scoring and independent holdout
evaluation rules before any query scoring.

- Primary: `TARGET_DISTRACTOR_MAX_MARGIN` = `T_max - D_max`
- Secondary diagnostic formulas (top-3 / centroid / medoid / mean)
- New holdout required; sample after refinement is not independent proof

No score rows, rankings, metrics, threshold, identity, or gallery mutation.

## Status

`COMPLETED_STAGE5D_F3H_TARGET_001_TARGET_DISTRACTOR_SCORING_AND_HOLDOUT_DESIGN_READY`

Exact next gate:
`STAGE5D-F3I_TARGET_001_NEW_INDEPENDENT_HOLDOUT_INGESTION_AND_PREFLIGHT`

## Run

```bash
conda run -n football-cv python scripts/run_reid_target_distractor_scoring_contract_and_holdout_design.py \
  --config configs/reid/target_distractor_scoring_contract_stage5d_target_001.yaml
```
