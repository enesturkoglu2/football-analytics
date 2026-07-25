# Stage 5D-B1A — Target 001 Manual Seed Selection Package

## Status

`COMPLETED_STAGE5D_B1A_TARGET_001_MANUAL_SEED_SELECTION_READY`

Exact next gate:
`STAGE5D-B1B_TARGET_001_MANUAL_SEED_SELECTION_FREEZE_AND_ANCHOR_DERIVATION`

## Purpose

Prepare a neutral visual package so a human can click/select the detection box
of **sarı takım 5 numaralı oyuncu** near a known video moment, without any
automatic identity suggestion.

This gate does **not**:

- fill or freeze a seed selection
- create anchors / gallery membership / prototypes / identity assignments
- run YOLO, ByteTrack, OSNet, PARSeq/OCR, or similarity inference

## Human seed window

- Representative seed frame: **290** (~9.67 s)
- Review window: frames **280–310** inclusive (~9.33–10.33 s)
- If frame 290 has no observation, nearest observation frame is reported with
  delta (in this rebuild, frame 290 has observations)

## Neutral codes

Every unique `(raw_track_id, segment_id)` observation chain in the window gets a
deterministic code:

`SEED_CANDIDATE_01`, `SEED_CANDIDATE_02`, …

Codes are stable across the window and carry no identity / kit / jersey / track
semantics on the visual panels. Machine-readable mapping retains lineage for a
later freeze gate.

## Review materials

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_manual_seed_selection_package`

- Sheet: `review_packages/target_001_manual_seed_selection/seed_selection_sheet_01.png`
- Clip: `review_packages/target_001_manual_seed_selection/target_001_manual_seed_window.mp4`

Visuals show only:

- full source frames
- existing person bboxes
- neutral `SEED_CANDIDATE_XX` labels
- frame index / time
- `HUMAN SEED REFERENCE` on the representative (or nearest) frame

Not shown: raw_track_id, segment_id, global ID, similarity, model identity, OCR,
auto jersey, target yes/no.

## Blank template

`templates/target_001_manual_seed_selection_template.csv`

- `selected_neutral_seed_code` starts empty
- all manual fields empty
- no model suggestion / prefilled target code

## Contract

- human click/box selection required
- existing bbox only; no interpolation
- no new detection/tracking/embedding/OCR/similarity/identity prediction
- no automatic selection / gallery / anchor freeze
- selected seed requires Stage 5D-B1B freeze + derivation
- unknown identity preserved

## Tracked files

Exact 4:

1. `scripts/run_reid_target_manual_seed_selection_package.py`
2. `configs/reid/target_manual_seed_selection_stage5d_target_001.yaml`
3. `tests/test_reid_target_manual_seed_selection_package.py`
4. `docs/setup/stage5d-target-manual-seed-selection-package.md`

## Exact next gate

`STAGE5D-B1B_TARGET_001_MANUAL_SEED_SELECTION_FREEZE_AND_ANCHOR_DERIVATION`

User confirms the neutral code; lineage is frozen; diversified anchors are
derived from that verified segment for further human review — still not final
gallery membership.
