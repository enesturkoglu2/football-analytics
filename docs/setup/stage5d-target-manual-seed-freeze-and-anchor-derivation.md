# Stage 5D-B1B — Manual Seed Selection Freeze and Anchor Derivation

## Status (this rebuild run)

`COMPLETED_STAGE5D_B1B_SEED_FROZEN_SOURCE_INELIGIBLE`

Exact next gate:
`STAGE5D-B1C_TARGET_001_ALTERNATE_ELIGIBLE_ANCHOR_SOURCE_REVIEW`

## Purpose

Freeze the human-approved neutral seed `SEED_CANDIDATE_07`, resolve its
existing track/segment/crop/embedding lineage, audit Stage 5D-A gallery
exclusion eligibility, and only then derive label-blind anchor review
candidates from that segment.

## Human-approved selection (immutable)

| Field | Value |
|---|---|
| selected_neutral_seed_code | `SEED_CANDIDATE_07` (exact string) |
| manual_target_confirmed | yes |
| manual_human_verified_number_seen | yes |
| manual_crop_valid | yes |
| manual_target_dominant | yes |
| human_verified_jersey_number | 5 |
| jersey_number_provenance | `human_visual_verification_not_automated_ocr` |
| reviewer / final_approver | Furkan |

OCR / similarity / alternate code suggestion are forbidden. Overlay “077”
artefacts are not canonical codes.

## Resolved lineage (dynamic from B1A mapping)

- raw_track_id / segment_id resolved from exact mapping row
- observation frames within review window 280–310
- bbox lineage audited against `segment_observations`
- existing OSNet embedding audited when present

## Eligibility branching

`selected_seed_source_eligibility` may be:

- `eligible_for_anchor_derivation`
- `frozen_identity_seed_only_stage5c_excluded`
- `frozen_identity_seed_only_leakage_excluded`
- `frozen_identity_seed_only_no_existing_embedding`
- `frozen_identity_seed_only_ambiguous_segment`
- `frozen_identity_seed_only_other_exclusion`

**If ineligible:** seed remains frozen as human-verified identity start;
derived anchors / contact sheets / gallery enrollment = 0.

**If eligible:** up to 8 deterministic diverse full-body crop candidates are
prepared for human review (still not gallery membership).

**If eligible but &lt;3 diverse candidates:** seed frozen; insufficient diversity
status; additional human seed window review next.

## This rebuild result

`SEED_CANDIDATE_07` → `raw_222_full` / track 222 is present in Stage 5C
`holdout_primary` (and track/segment exclusion keys). Therefore:

- seed freeze published
- eligibility = `frozen_identity_seed_only_stage5c_excluded`
- derived anchors = 0
- contact sheet PNG = 0
- gallery / prototypes / identity = 0

## Artifacts

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_seed_freeze_anchor_derivation`

- `seed_freeze/target_001_manual_seed_selection_frozen.json`
- `eligibility/target_001_selected_seed_source_eligibility.json`
- inventory / templates (empty derivation when ineligible)
- `stage5d_b1b_summary.json` / `stage5d_b1b_manifest.json`

## Tracked files

Exact 4:

1. `scripts/run_reid_target_manual_seed_freeze_anchor_derivation.py`
2. `configs/reid/target_manual_seed_freeze_anchor_derivation_stage5d_target_001.yaml`
3. `tests/test_reid_target_manual_seed_freeze_anchor_derivation.py`
4. `docs/setup/stage5d-target-manual-seed-freeze-and-anchor-derivation.md`

## Exact next gate

Because source is Stage 5C excluded:

`STAGE5D-B1C_TARGET_001_ALTERNATE_ELIGIBLE_ANCHOR_SOURCE_REVIEW`

Alternate sources must remain linkable by human video continuity to the frozen
seed identity, while avoiding Stage 5C / evaluation exclusion.
