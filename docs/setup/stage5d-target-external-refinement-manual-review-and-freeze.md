# Stage 5D-F3D — External Refinement Manual Review and Freeze

## Purpose

Freeze exact human decisions from the F3C external refinement review
package:

- 11 target-view candidates (2 expansion yes, 7 redundant valid, 2 ambiguous)
- 135 unreviewed external occurrences
- 1 additional target occurrence (`EXT_161`)
- 35 same-team distractor sources (not hard-negative members yet)
- special `EXT_213` target-present / multi-person ambiguous record

No embeddings, gallery-v2, hard-negative gallery, OCR/similarity,
crop regeneration, or sample access.

## Transcription corrections

Five F3C inventory codes were corrected from an earlier prompt draft
(human transcription only; not model matching):

| From | To | Decision |
|---|---|---|
| EXT_016 | EXT_019 | other_team_player |
| EXT_048 | EXT_049 | multi_person_ambiguous |
| EXT_065 | EXT_066 | multi_person_ambiguous |
| EXT_080 | EXT_082 | other_team_player |
| EXT_228 | EXT_226 | other_team_player |

## Status

`COMPLETED_STAGE5D_F3D_TARGET_001_EXTERNAL_REFINEMENT_DECISIONS_FROZEN`

Exact next gate:
`STAGE5D-F3E_TARGET_001_EXTERNAL_REFINEMENT_CROP_QUALITY_AND_HARD_NEGATIVE_REVIEW_PACKAGE`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_refinement_manual_freeze.py \
  --config configs/reid/external_refinement_manual_freeze_stage5d_target_001.yaml
```
