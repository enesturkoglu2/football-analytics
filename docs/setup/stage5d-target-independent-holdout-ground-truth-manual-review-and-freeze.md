# Stage 5D-F3L — Independent holdout v2 ground-truth manual review and freeze

## Purpose / Amaç

Freeze exact human ground-truth decisions for the 141 `H2_GT_REVIEW_000001`…`000141`
items from the F3K review package. No gallery, OSNet, similarity, scoring, ranking,
metrics, threshold, identity assignment, or gallery mutation.

F3K review paketindeki 141 review item için Furkan onaylı kararları immutable
biçimde dondurur.

## Expected distribution

| Decision | Count |
|---|---|
| target_occurrence_yes | 10 |
| target_occurrence_no (same-team) | 55 |
| target_occurrence_no (other-team) | 50 |
| non_player | 5 |
| invalid | 7 |
| multi_person_ambiguous | 14 |
| uncertain | 0 |
| clean positive | 10 |
| clean negative | 110 |
| clean same-team negative | 55 |
| reviewed metric excluded | 21 |
| unreviewed ineligible | 102 |
| complete universe | 243 |

## Clean positives

`H2_GT_REVIEW_000010, 000030, 000061, 000090, 000094, 000104, 000124, 000129, 000135, 000136`

All positives: jersey `5`, provenance `human_visual_review_by_Furkan`.

## Component policy

`ONE_FROZEN_SEGMENT_PER_COMPONENT_NO_CROSS_TRACK_LINK_EVIDENCE`

- Reviewed: `H2_GT_COMPONENT_<review six digits>`
- Ineligible: `H2_GT_COMPONENT_INELIG_<segment six digits>`
- 243 unique components, conflict=0

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_independent_holdout_v2_ground_truth_manual_freeze.py \
  --config configs/reid/independent_holdout_v2_ground_truth_manual_freeze_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F3L_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_FROZEN`

Readiness:
`TARGET_001_INDEPENDENT_HOLDOUT_V2_READY_FOR_FROZEN_QUERY_EMBEDDING_AND_SCORING`

Exact next gate:
`STAGE5D-F3M_TARGET_001_NEW_INDEPENDENT_HOLDOUT_OSNET_QUERY_EMBEDDING_AND_FROZEN_TARGET_DISTRACTOR_SCORING`

## Tests

```bash
conda run -n football-cv python -m unittest discover -s tests \
  -p 'test_reid_independent_holdout_v2_ground_truth_manual_freeze.py' -v
```
