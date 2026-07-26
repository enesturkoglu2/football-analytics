# Stage 5D-F3J — Independent holdout v2 label-blind detection, tracking, segment universe

## Purpose

Build a label-blind person detection / ByteTrack / objective segment universe on the
accepted independent holdout v2 clip only. Prepare an immutable universe for the next
similarity-blind human ground-truth review package (F3K).

## Status

- Final: `COMPLETED_STAGE5D_F3J_TARGET_001_NEW_INDEPENDENT_HOLDOUT_LABEL_BLIND_UNIVERSE_BUILT`
- Readiness: `TARGET_001_INDEPENDENT_HOLDOUT_V2_LABEL_BLIND_UNIVERSE_READY_FOR_GROUND_TRUTH_REVIEW_PACKAGE`
- Next: `STAGE5D-F3K_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_REVIEW_PACKAGE`

## Inputs (immutable)

- Holdout: `data/test_clips/target_001_independent_holdout_v2.mp4`
  - SHA-256 `bbfe3669cfbf39534f71a80131401bb4d9f931c7a4ea485404ab2fa207a6231f`
  - 1058 frames, 1336×754, 30 fps
- Detector: `models/yolo11n.pt` (frozen B1E-B contract)
- Tracker: `configs/tracking/bytetrack_stage3.yaml` (frozen B1E-B contract)
- Segmentation: Stage 5B3 purity/policy with `automatic_track_split_enabled=false` and
  null split thresholds → pass-through segment per raw track

## Forbidden in this gate

- sample.mp4 / external enrollment video reads
- gallery-v1/v2 / distractor embeddings, centroids, cross-similarity
- OSNet / PARSeq / OCR / team classifier
- crop export, embeddings, similarity, ranking, GT labels, metrics, threshold selection
- identity / jersey / team assignment

## Pipeline

1. Validate Git + F3I acceptance + F3I snapshot + holdout SHA/metadata
2. Resolve frozen detector/tracker from B1E-B; write pre-inference contract
3. Resolve frozen segmentation contract; write pre-build contract
4. Pass-1: decode → YOLO person detect → ByteTrack → pass-through segments → eligibility
5. Pass-2: independent full replay (does not reuse Pass-1 detections as tracker input)
6. Exact determinism compare; atomic publish under
   `outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe`

## Stable IDs

- Detection: `H2_DET_<frame6>_<rank3>` (x1,y1,x2,y2 asc; confidence desc; detector index)
- Raw track: `H2_RAW_<id6>`; observation: `H2_OBS_<frame6>_<track6>`
- Segment: `H2_SEG_<seq6>` (raw track id, then start/end frame)

## Review eligibility

Objective only (`min_observation_count=3`, valid in-frame bbox). Ineligible segments remain
in the complete universe. Representative frame candidates are metadata only (no crops).

## Run

```bash
conda run -n football-cv python scripts/run_reid_independent_holdout_v2_label_blind_universe.py \
  --config configs/reid/independent_holdout_v2_label_blind_universe_stage5d_target_001.yaml
```

## Tests

```bash
conda run -n football-cv python -m unittest tests.test_reid_independent_holdout_v2_label_blind_universe -v
```
