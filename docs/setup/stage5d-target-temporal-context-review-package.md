# Stage 5D-B1 — Target 001 Temporal Context Review Package

## Status

`COMPLETED_STAGE5D_B1_TARGET_001_TEMPORAL_CONTEXT_REVIEW_READY`

Exact next gate:
`STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE`

## Purpose

Prepare temporal visual context for the nine Stage 5D-B label-blind anchor
candidates so a human can recognize continuity of
**sarı takım 5 numaralı oyuncu** across neighboring frames.

This gate does **not**:

- write or freeze manual anchor decisions
- create gallery membership / prototypes / identity assignments
- run YOLO, ByteTrack, OSNet, PARSeq/OCR, or similarity inference
- filter/rank by kit color, jersey number, similarity, or model identity

## Upstream Stage 5D-B package

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_anchor_review_package`

Validated:

- target_001 frozen; human jersey provenance only
- eligible candidates = 9 (`target_001_anchor_001` … `_009`)
- Stage 5D-B annotation template blank
- gallery / prototypes / identity = 0

## Temporal window

For each candidate:

- center = Stage 5D-B representative frame
- preferred span ≈ ±60 frames (±2 s at 30 fps)
- hard-clamped to the candidate segment observation range
- no expansion into Stage 5C / excluded temporal windows
- no bbox interpolation on frames without observations

Deterministic sheet observations (≤ 7, duplicates collapsed):

1. nearest to context start  
2. ~rep−40  
3. ~rep−20  
4. representative / nearest  
5. ~rep+20  
6. ~rep+40  
7. nearest to context end  

## Visual materials

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_temporal_context_review_package`

Contact sheets (3):

- `temporal_contact_sheet_01.png` → anchors 001–003  
- `temporal_contact_sheet_02.png` → anchors 004–006  
- `temporal_contact_sheet_03.png` → anchors 007–009  

Each candidate is one horizontal row. Each panel shows:

- frame index + video time  
- REP / CTX marker  
- full-frame context with **only** the candidate’s upstream bbox  
- zoomed person view (fixed padding ratio; visualization only)

Not shown: raw_track_id, global ID, similarity, embedding distance, model
identity, OCR prediction/confidence, expected jersey overlay, yes/no suggestion.

Clips (9):

`review_packages/target_001_temporal_context_review/clips/target_001_anchor_XXX_context.mp4`

- segment-bounded context window  
- source frame order preserved  
- bbox drawn only on observation frames  
- no audio; fps = source fps  

## Annotation template

New blank template (Stage 5D-B template is **not** mutated):

`templates/target_001_temporal_context_review_template.csv`

Allowed `temporal_review_decision` values match Stage 5D-B vocabulary.
Tri-state fields: `yes` / `no` / `uncertain`.
`manual_human_verified_number_seen_in_context` is a human observation field,
not OCR.

## Contract highlights

- no new detection / tracking / embedding / OCR / similarity / identity prediction  
- bbox source = segmented ReID `segment_observations.source_observation.bbox_xyxy`  
- zoom padding recorded; zoom is not a gallery/embedding artifact  
- human approval required for any later freeze  
- manual decisions / approved anchors / gallery / prototypes / identity = 0  

## Tracked files

Exact 4:

1. `scripts/run_reid_target_temporal_context_review_package.py`  
2. `configs/reid/target_temporal_context_review_stage5d_target_001.yaml`  
3. `tests/test_reid_target_temporal_context_review_package.py`  
4. `docs/setup/stage5d-target-temporal-context-review-package.md`  

README / PROJECT_CONTEXT / Stage 5D-B tracked files unchanged.

## Exact next gate

`STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE`

In B2, humans review temporal sheets/clips and enter decisions; freeze requires
final approval; `target_anchor_yes` is still not final gallery membership.
