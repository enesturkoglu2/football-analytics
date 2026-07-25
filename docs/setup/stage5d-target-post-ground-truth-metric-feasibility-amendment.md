# Stage 5D-F2B — Post-Ground-Truth Metric Feasibility Amendment

## Purpose

Amend the F1 strong-signal outcome before F3 scoring so segment
`Recall@5=1.0` is replaced by mathematically attainable ceilings
derived only from frozen ground-truth counts (8 positives).

Does not load embeddings, compute similarity/rank, mutate ground
truth or gallery, or select a threshold.

## Frozen support

| Unit | Count |
|---|---|
| segment positives | 8 |
| segment negatives | 110 |
| segment excluded | 32 |
| clean positive components | 4 |
| clean negative components | 95 |
| excluded components | 26 |
| conflicting components | 4 |

## Amended strong-signal (reachable)

- segment Recall@5 = 0.625 (5/8 ceiling)
- segment Recall@10 = 1.0
- component Recall@5 = 1.0
- segment/component AP ≥ 0.80
- min positive score > max negative score (segment and component)

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_post_gt_metric_feasibility_amendment.py \
  --config configs/reid/post_gt_metric_feasibility_amendment_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F2B_TARGET_001_METRIC_CONTRACT_AMENDED_READY_FOR_SCORING`

Exact next gate:
`STAGE5D-F3_TARGET_001_INDEPENDENT_SAMPLE_RETRIEVAL_SCORING_AND_EVALUATION`
