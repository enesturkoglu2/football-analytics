# Stage 5D-B1C2 — Target 001 Seed to Eligible Source Temporal Bridge Review

## Purpose

Show frozen-but-enrollment-ineligible `SEED_CANDIDATE_07` as a human
continuity reference across frames 280–420, and surface only Stage 5D
enrollment-eligible existing segments in that window as neutral
`BRIDGE_CANDIDATE_*` codes for later human continuation selection.

This gate does **not** freeze a bridge selection, derive anchors, or create
gallery membership.

## Frozen seed visual policy

| Item | Value |
|---|---|
| Code | `SEED_CANDIDATE_07` |
| Segment / track | `raw_222_full` / 222 |
| Visual label | `FROZEN_HUMAN_SEED_REF` |
| Warning | `NOT ENROLLABLE — HUMAN CONTINUITY REFERENCE ONLY` |
| Embedding used | **false** |
| Enrollment / scoring / gallery | **forbidden** |

## Bridge window

- Frames **280–420** inclusive (~9.33–14.00 s)
- Covers frozen seed visibility (~290) and Stage 5D-B candidate region (~394–397)
- Existing observations only; no bbox interpolation; no new tracking

## Eligible bridge candidates

A non-frozen chain is eligible only if it passes Stage 5C / Stage 5D-A
exclusion keys and has verified existing OSNet embedding + crop lineage.
Stage 5D-B `target_001_anchor_00X` lineage is reported in machine-readable
mapping only — never shown as identity proof on visuals.

## Artifacts

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_seed_to_eligible_bridge_review`

- 3 contact sheets + 1 bridge MP4
- blank CSV template (one row per bridge candidate)
- eligibility audit / contract / summary

## Statuses

Eligible > 0:

`COMPLETED_STAGE5D_B1C2_SEED_TO_ELIGIBLE_BRIDGE_REVIEW_READY` →
`STAGE5D-B1D_TARGET_001_BRIDGE_SOURCE_SELECTION_FREEZE_AND_ANCHOR_DERIVATION`

Eligible = 0:

`COMPLETED_STAGE5D_B1C2_NO_ELIGIBLE_BRIDGE_SOURCE` →
`STAGE5D-B1E_TARGET_001_EXTERNAL_ENROLLMENT_CLIP_DESIGN`

## Run

```bash
conda run -n football-cv python scripts/run_reid_target_seed_to_eligible_bridge_review.py \
  --config configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml
```
