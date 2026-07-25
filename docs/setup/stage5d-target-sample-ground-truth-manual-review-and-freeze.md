# Stage 5D-F2A — Sample Ground-Truth Manual Review and Freeze

## Purpose

Freeze exact human ground-truth decisions for the 150 `SAMPLE_EVAL`
items from the F2 review package. No similarity, ranking, threshold,
identity assignment, or gallery mutation.

## Expected distribution

| Decision | Count |
|---|---|
| target_occurrence_yes | 8 |
| target_occurrence_no | 103 |
| non_player | 7 |
| uncertain | 8 |
| multi_person_ambiguous | 24 |
| invalid | 0 |
| clean positive metric | 8 |
| clean negative metric | 110 |
| excluded metric | 32 |

## Clean positives

`SAMPLE_EVAL_003, 024, 028, 042, 046, 069, 100, 102`

Target-present ambiguous (metric-excluded): `108`, `148`

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_sample_ground_truth_manual_freeze.py \
  --config configs/reid/sample_ground_truth_manual_freeze_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F2A_TARGET_001_SAMPLE_GROUND_TRUTH_FROZEN`

Exact next gate:
`STAGE5D-F3_TARGET_001_INDEPENDENT_SAMPLE_RETRIEVAL_SCORING_AND_EVALUATION`
