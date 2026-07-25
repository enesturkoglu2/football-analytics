# Stage 5D-B1D — Bridge Review No-Selection Freeze and External Enrollment Handoff

## Status

`COMPLETED_STAGE5D_B1D_NO_BRIDGE_SELECTION_EXTERNAL_ENROLLMENT_REQUIRED`

Exact next gate:
`STAGE5D-B1E_TARGET_001_EXTERNAL_ENROLLMENT_CLIP_DESIGN_AND_INGEST`

## Purpose

Freeze the human bridge-review outcome that **no** eligible bridge candidate
continues the frozen `SEED_CANDIDATE_07` target identity in the current video,
close current-video eligible-source search, and publish external enrollment
clip requirements.

## Frozen human decision

| Field | Value |
|---|---|
| selected_bridge_candidate_code | *(empty)* |
| manual_target_continuation_found | no |
| manual_review_result | `NO_ELIGIBLE_BRIDGE_CONTINUATION_SELECTED` |
| reviewer / final_approver | Furkan |

Per-candidate decisions:

| Code | Decision |
|---|---|
| BRIDGE_CANDIDATE_01 | non_player |
| BRIDGE_CANDIDATE_02 | target_anchor_no |
| BRIDGE_CANDIDATE_03 | target_anchor_no |
| BRIDGE_CANDIDATE_04 | target_anchor_no |
| BRIDGE_CANDIDATE_05 | target_anchor_no |

Force selection is forbidden. Original frozen seed remains enrollment-ineligible.

## Artifacts

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_bridge_review_no_selection_freeze`

- `bridge_review_freeze/target_001_bridge_review_decisions_frozen.csv`
- `bridge_review_freeze/target_001_bridge_review_no_selection.json`
- `bridge_review_freeze/target_001_bridge_review_freeze_contract.json`
- `bridge_review_freeze/target_001_bridge_review_freeze_manifest.json`
- `external_enrollment_handoff/target_001_external_enrollment_requirements.json`
- `stage5d_b1d_summary.json` / `stage5d_b1d_manifest.json`

## External enrollment requirements (handoff)

- enrollment-only clip; not future evaluation input
- human jersey 5 verified; automated OCR=false
- human seed selection + manual frozen enrollment
- automatic gallery growth=false
- unknown identity preserved

## Run

```bash
conda run -n football-cv python scripts/run_reid_target_bridge_review_no_selection_freeze.py \
  --config configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml
```
