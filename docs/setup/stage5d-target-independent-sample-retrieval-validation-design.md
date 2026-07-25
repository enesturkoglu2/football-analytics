# Stage 5D-F1 — Independent Sample Retrieval Validation Design

## Purpose

Freeze the independent retrieval validation protocol for `target_001`
**before** any sample.mp4 similarity scoring, ranking, ground-truth fill,
or threshold selection.

This gate:

- validates gallery-v1 readiness and immutability
- reconfirms enrollment/evaluation independence (overlap=0)
- inventories the sample scoreable ReID universe (150 / 141)
- freezes label-blind ground-truth, scoring, metric, and outcome contracts

This gate does **not**:

- run OSNet on sample.mp4
- compare sample embeddings to the gallery
- produce similarity or ranking rows
- fill human ground-truth decisions
- select thresholds or assign identities
- mutate or grow the gallery

## Expected counts at gate close

| Item | Value |
|---|---|
| gallery members | 7 |
| sample ground-truth decisions | 0 |
| sample similarity rows | 0 |
| retrieval rankings | 0 |
| threshold selected | false |
| identity assignments | 0 |
| automatic gallery growth | false |

## Primary scoring formula (preregistered)

`max_individual_cosine` = max cosine vs the 7 individual frozen gallery embeddings.

Secondary diagnostics only:

- `top3_mean_individual_cosine` (k=3 frozen before seeing scores)
- `centroid_cosine`
- `medoid_cosine`
- `mean_individual_cosine`

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_target_independent_validation_design.py \
  --config configs/reid/target_independent_validation_design_stage5d_target_001.yaml
```

## Output root

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_validation_design`

## Success status

`COMPLETED_STAGE5D_F1_TARGET_001_INDEPENDENT_VALIDATION_DESIGN_READY`

Exact next gate:
`STAGE5D-F2_TARGET_001_SAMPLE_GROUND_TRUTH_REVIEW_PACKAGE`
