# Stage 5D-F3M — Independent Holdout OSNet Query Embedding and Frozen Target–Distractor Scoring

## Purpose

Run a single, sealed evaluation of Target 001 independent holdout v2:

1. Freeze and verify F3L ground truth, F3H scoring contract, F3G galleries, and F3K crops.
2. Build a label-blind scoreable query projection of exact 115 clean-player segments.
3. Embed queries with canonical OSNet (`sn-reid-cpu`) in two independent passes.
4. Compute frozen target/distractor cosine matrices and `TARGET_DISTRACTOR_MAX_MARGIN`.
5. Rank label-blind, seal execution, then join F3L GT and compute preregistered metrics.
6. Apply F3H outcome rules without threshold/identity/gallery mutation.

## Official universe

| Cohort | Count |
|--------|------:|
| Complete holdout | 243 |
| Scoreable clean-player queries | 115 |
| Clean positives | 10 |
| Clean player negatives | 105 |
| Same-team negatives | 55 |
| Other-team negatives | 50 |
| Scoring exclusions | 128 |

Non-player reviewed items remain clean-negative for GT provenance but are **not** scored.

## Commands

```bash
cd /home/enesturkoglu2/projects/football-analytics

conda run -n sn-reid-cpu python \
  scripts/run_reid_independent_holdout_v2_frozen_scoring_evaluation.py \
  --config configs/reid/independent_holdout_v2_frozen_scoring_stage5d_target_001.yaml

conda run -n football-cv python -m unittest \
  tests.test_reid_independent_holdout_v2_frozen_scoring_evaluation -v
```

## Hard constraints

- No holdout MP4 decode; reuse F3K representative crops only.
- No gallery member add/remove/reorder/reweight.
- No F3H formula / aggregation / tie-break / metric / outcome mutation.
- No F3L GT decision mutation.
- GT labels hidden until pre-GT join seal.
- No threshold selection, identity assignment, enrollment, or hard-negative mining.
- Offline only; no package/environment changes.

## Primary score

```
T_max(q) = max cosine(q, 13 target gallery-v2 members)
D_max(q) = max cosine(q, 23 distractor gallery-v1 members)
S_primary(q) = T_max(q) - D_max(q)
```

Tie-break: `S_primary` ↓, `T_max` ↓, `D_max` ↑, stable query ID ↑.

## Outputs

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_frozen_target_distractor_scoring_evaluation`

## Next gate

`STAGE5D-F3N_TARGET_001_INDEPENDENT_HOLDOUT_RESULT_AUDIT_AND_ERROR_ANALYSIS_PACKAGE`
