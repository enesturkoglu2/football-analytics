# Stage 5C — Clean label-blind discovery / holdout design

- **Canonical generation:** `r2_capacity_balanced`
- **Canonical root:**
  `outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced`
- **Design freeze gate:** REBUILD-R4D
- **Discovery annotation:** REBUILD-R5 complete (reserve closed)
- **Discovery PARSeq candidate gate:** REBUILD-R6 complete
  (cut `0.99992299168434329`; support=1; not deployment)
- **Exact next gate:**
  `REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE`

This design is **label-blind** and **prediction-blind**. It does not
reuse the historical 78-item pilot as an annotation set and does not
reuse C3D/C3E membership or any C3E threshold.

## 1. Clean label-blind 474 universe

Source:

`outputs/reid/full_stage4b_rebuild_r2_stage5c_visibility_universe`

Contract:

- total items = **474**
- reused_baseline_selected_crop = **399**
- recomputed_manual_segment = **75**
- all `annotation_status=unreviewed`
- manual fields blank; OCR/prediction/confidence absent
- ROI-valid only

Builder / config / tests (canonical aktif):

- `scripts/build_reid_jersey_clean_review_universe.py`
- `configs/reid/jersey_clean_review_universe_stage5c_rebuild_r2.yaml`
- `tests/test_reid_jersey_clean_review_universe.py`

## 2. Feature / strata contract

Composite score uses only model-independent geometric / visibility /
quality / join-missingness features (no jersey labels, no OCR).

Strata (sampling tools only; **not** ground truth):

| Stratum | Count |
|---|---|
| high_signal_candidate | 190 |
| mid_signal_candidate | 166 |
| safety_candidate | 118 |

Semantics:

- high ≠ readable positive
- safety ≠ negative ground truth

## 3. Leakage group contract

Hard zero-overlap across batches for:

- review_item_id / crop / path / SHA
- segment_id
- raw_track_id
- documented_global_candidate_id
- leakage_group_id
- near_duplicate_cluster_id

Also:

- transitive closure
- documented accepted components co-located
- no raw-track / segment / component cross-batch leakage

Universe leakage groups ≈ **129** (exact value in split audit).

## 4. Near-duplicate contract

- algorithm: dHash
- near_duplicate_max_distance = **8**
- edges / clusters recorded in leakage audit
- near-duplicate clusters do not cross batches

## 5. Initial R4 imbalance

First clean split (`…_stage5c_clean_split`, generation historically
unbalanced on source type):

- discovery_primary had mixed reused/recomputed
- holdout batches contained **0 recomputed**
- root kept immutable for provenance

Deprecation for downstream applies **only after** successful
capacity-balanced publish (R4C).

## 6. Blocked R4B (21-item exact quota)

Requested selected recomputed = **21** under exact batch source
quotas failed:

- dynamically measured maximum selectable recomputed = **18**
- status: `BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY`
- leakage/diversity rules were **not** relaxed
- no balanced final root published in R4B

## 7. R4C capacity audit

Recomputed capacity under fixed leakage + diversity:

- recomputed items = 75
- recomputed raw tracks / leakage groups = 13 / 13
- per-track limit ≤2 and frame-gap ≥60 ⇒
  **global maximum selectable recomputed = 18**

Hard minima (every batch must represent recomputed):

- discovery_primary ≥ 4
- discovery_reserve ≥ 1
- holdout_primary ≥ 4
- holdout_reserve ≥ 1

## 8. Maximum feasible recomputed and only feasible vector

- `maximum_feasible_recomputed_count = 18`
- `selected_recomputed_is_maximum_feasible = true`
  (exhaustive descending search under fixed allocator)
- Hamilton / proportional reference for 18: **6 / 2 / 7 / 3**
  (not feasible under allocator + leakage/diversity)
- **Only feasible vector at total 18:** **5 / 7 / 4 / 2**

## 9. Four batch distributions

| Batch | total | high/mid/safety | reused | recomputed |
|---|---|---|---|---|
| discovery_primary | 40 | 20/10/10 | 35 | 5 |
| discovery_reserve | 16 | 8/4/4 | 9 | 7 |
| holdout_primary | 48 | 24/12/12 | 44 | 4 |
| holdout_reserve | 24 | 12/6/6 | 22 | 2 |
| **selected** | **128** | — | **110** | **18** |
| **unselected** | **346** | — | **289** | **57** |

Allocated = 128; unselected = 346.

## 10. Overlap = 0

Canonical split overlap/leakage checks: **0** on all hard keys.

Batch-internal diversity:

- same segment selected ≤ 1
- same raw track selected ≤ 2
- same-track frame gap ≥ 60
- deterministic ordering (no random)

## 11. Historical-never-seen limitation

- `strict_historical_never_seen_claim = false`
- Rebuild is independent **within the r2 protocol**, not a claim that
  no historical human ever saw related crops in older pilots

## 12. Player-level independence limitation

- `player_identity_independence_guaranteed = false`
- Documented global candidates / raw tracks are **not** proven
  player identities
- Leakage controls are operational (track/segment/component/near-dup),
  not person-level guarantees

## 13. Canonical balanced root

```text
outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced
canonical_split_generation = r2_capacity_balanced
```

Builder / config / tests (canonical aktif):

- `scripts/build_reid_jersey_clean_split.py`
- `configs/reid/jersey_clean_split_stage5c_rebuild_r2.yaml`
- `tests/test_reid_jersey_clean_split.py`

## 14. Previous R4 root deprecated

Immutable previous root:

`outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split`

After successful R4C publish:

- `previous_r4_split_deprecated_for_downstream = true`
- reason:
  `source_type_distribution_not_representative_in_holdout`

Do not use the previous R4 root for new annotation.

## 15. Threshold / labels / predictions (design-time vs R6)

At clean-split design / R4–R5 time, preregistration remained:

- `threshold_selected = false` on the split package itself
- discovery/holdout labels and predictions unseen for split design
- historical threshold reused = false

After REBUILD-R6, a **discovery-derived holdout-validation candidate
cut** exists in the separate gate root:

- exact decimal `0.99992299168434329`
- float64 hex `3fefff5e8079b000`
- operator `>=`
- support=1 (`discovery_primary_028`)
- `deployment_threshold_selected=false`
- `mutable_after_holdout=false`
- historical C3E cut **not** used

Negative digit emission on discovery negatives was **27/27** under
the frozen recognizer-only contract; safety relies on the cut, not
on zero emission.

Details:
[stage5c-discovery-parseq-candidate-gate-r2.md](stage5c-discovery-parseq-candidate-gate-r2.md)

## 16. Preregistered discovery / holdout rules

Discovery primary minima (labels, after annotation gate):

- readable positive ≥ 8
- non-readable / negative ≥ 20

Holdout primary minima:

- readable positive ≥ 10
- non-readable / negative ≥ 24

Reserve opening:

- discovery reserve only if primary minima unmet (batch_order only)
- holdout reserve only if primary label minima unmet
- no inference before reserve decision / freeze rules as preregistered

Holdout decision taxonomy (PASS / INCONCLUSIVE / FAIL / BLOCKED)
remains immutable after results.

## 17. Contact sheets

- discovery primary ≤ 4 PNG
- discovery reserve ≤ 2 PNG
- holdout primary ≤ 4 PNG
- holdout reserve ≤ 2 PNG
- total ≤ 12 PNG; JPEG copies = 0; MP4 = 0
- reviewer-facing sheets omit stratum/score/source-type/track/segment/
  documented-global/quality/kit/near-dup metadata

## 18. Superseded C3F-A code (retained)

Preserved but superseded holdout-design triad:

- `scripts/build_reid_jersey_parseq_holdout.py`
- `configs/reid/jersey_parseq_holdout_stage5c_c3f_a.yaml`
- `tests/test_reid_jersey_parseq_holdout.py`

- `superseded_by` = canonical capacity-balanced split root above
- `must_not_be_used_for_new_r2_annotation = true`
- `retained_for_history_and_tests = true`

## 19. Exact next gate

`REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE`

Until then / during R7:

- do not retune the frozen discovery candidate cut after holdout
  results
- do not open/review holdout packages before the approved gate
- do not treat the candidate cut as a deployment threshold
- discovery reserve remains closed for threshold search
