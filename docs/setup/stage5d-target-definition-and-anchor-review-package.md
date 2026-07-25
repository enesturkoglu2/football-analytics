# Stage 5D-B — Target Definition and Anchor Review Package

## Status

`COMPLETED_STAGE5D_B_TARGET_001_ANCHOR_REVIEW_PACKAGE_READY`

Exact next gate:
`STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE`

## Purpose

Freeze the first human-approved target definition (`target_001`) and publish a
label-blind, deterministic full-body anchor review package so a reviewer can
mark candidate images manually.

This gate does **not**:

- create gallery membership
- freeze approved anchors (that is Stage 5D-B2)
- run similarity retrieval / candidate ranking
- run PARSeq / OCR / YOLO / OSNet inference
- assign identities or merge tracks
- open Stage 5C discovery/holdout reserves for labeling

## Stage 5C closure context

Stage 5C remains closed:

- holdout validation: `INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT`
- automated PARSeq jersey evidence: disabled
- Stage 5E automated jersey channel: `diagnostic_only`
- appearance ReID remains primary
- unknown identity preserved

Stage 5C discovery/holdout items are **not** anchor sources. They are readable
only for exclusion / leakage provenance.

## Stage 5D-A preflight context

Preflight root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_gallery_preflight`

Validated before package publish:

- 150 embedded / 141 no-embedding OSNet segments
- embedding dim 512; NaN/Inf/zero-vector = 0
- blank target template
- gallery members / prototypes / identity assignments = 0
- automatic gallery growth = false
- unknown identity preserved = true

## Target definition (human-approved)

Frozen artifact:

`target_definition/target_001_definition_frozen.json`

| Field | Value |
|---|---|
| target_id | `target_001` |
| target_alias | sarı takım 5 numaralı oyuncu |
| identity_basis | `human_visual_verification_from_source_video` |
| human_verified_jersey_number | 5 |
| jersey_number_provenance | `human_verified_by_user_not_automated_ocr` |
| reviewer / final_approver | Furkan |
| target_definition_frozen | true |
| automated_jersey_used | false |

Policy:

- Jersey `5` is **not** an automated OCR result.
- Jersey metadata is explanatory only; it is not identity assignment.
- Anchor membership requires separate visual human decisions.
- After freeze, alias/basis are immutable within this review turn.

## Upstream segmented ReID provenance

Canonical Stage 5 replay segmented ReID (rebuild r2):

- final segment plan: 291
- embedded segments with existing OSNet 512-D vectors: 150
- no-embedding segments: 141 (excluded; no recompute)

Only the 150 embedded units enter the candidate universe.

## Gallery / evaluation leakage exclusions

Stage 5D-A preregistered overlap keys are applied against Stage 5C clean-split
batches (`discovery_primary`, `discovery_reserve`, `holdout_primary`,
`holdout_reserve`):

- segment_id
- raw_track_id
- crop_id
- source crop path / SHA
- frame identity
- exact duplicate group (`leakage_group_id`)
- near-duplicate component
- documented-link component
- temporal source window (`timeline_bin`)

Eligible count after exclusion is reported dynamically in the package summary.
If eligible count were 0, the gate would block with
`BLOCKED_STAGE5D_B_NO_ELIGIBLE_ANCHOR_SOURCE`.

## Representative full-body crop policy

For each eligible segment, exactly one existing crop is selected:

- identity-blind
- deterministic
- existing crops only (baseline reused or recomputed segment crops)
- no new YOLO/OSNet inference or crop regeneration

Preference order (metadata + deterministic image measurements only):

1. full-visibility proxy (bbox / short side)
2. crop area
3. edge margin (truncation proxy)
4. image size
5. Laplacian blur proxy
6. near segment mid-time

Forbidden selection signals: jersey number, kit-color target filter, similarity
score, embedding-based target prediction, tracker/global ID as identity proof.

## Review package

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_anchor_review_package`

Contact sheets:

- `review_packages/target_001_anchor_review/contact_sheet_XX.png`
- ≤ 12 items / sheet
- deterministic order
- visual panel shows only: order, `anchor_candidate_id`, full-body crop

Not shown on sheets: similarity, embedding distance, model identity, PARSeq/OCR,
OCR confidence, expected jersey overlay, global ID result, target-positive
prediction. Track/segment lineage stays in machine-readable manifests only.

## Annotation template

`templates/target_001_anchor_review_annotation_template.csv`

Allowed `manual_anchor_decision` values:

- `target_anchor_yes`
- `target_anchor_no`
- `uncertain`
- `invalid`
- `multi_person_ambiguous`
- `non_player`

All manual fields start blank. Only `target_anchor_yes` may later become a
frozen anchor (Stage 5D-B2). Even `target_anchor_yes` is **not** gallery
membership in Stage 5D-B.

## Anchor review contract highlights

- no automatic enrollment
- no similarity ranking
- no OCR usage
- no gallery membership / prototypes / identity assignment
- human review + final approval required
- anchor freeze requires Stage 5D-B2 (or later dedicated freeze gate)
- unknown identity preserved

At Stage 5D-B completion:

- manual decisions = 0
- approved anchors = 0
- gallery members = 0
- prototypes = 0
- identity assignments = 0

## Tracked files for this gate

Exact 4:

1. `scripts/run_reid_target_anchor_review_package.py`
2. `configs/reid/target_anchor_review_stage5d_target_001.yaml`
3. `tests/test_reid_target_anchor_review_package.py`
4. `docs/setup/stage5d-target-definition-and-anchor-review-package.md`

README / PROJECT_CONTEXT are intentionally unchanged until human annotation
completes.

## Limitations

- Package supports manual review only; no enrollment freeze yet.
- No-embedding segments remain out of scope (no recompute).
- Stage 5C jersey channel stays diagnostic-only.
- Similarity-based candidate expansion is deferred to Stage 5D-C after frozen
  anchors exist.

## Exact next gate

`STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE`

In B2:

- review only the published `target_001` contact sheets
- mark each candidate with the allowed decision vocabulary
- no freeze without user final approval
- no similarity retrieval
- no final gallery membership
- no automated jersey evidence
