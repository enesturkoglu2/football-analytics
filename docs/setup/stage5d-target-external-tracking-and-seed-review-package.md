# Stage 5D-B1E-B — External Clip Detection, Tracking, and Seed Review Package

## Purpose

Run canonical YOLO person detection once on the enrollment-only external clip
(frames 0–783), replay ByteTrack twice from saved detections for determinism,
assign neutral `EXT_###` codes, and publish a human seed-review package.

This gate does **not** select targets, compute OSNet embeddings, run OCR, or
create gallery/anchors/identity.

## Canonical settings

| Item | Value |
|---|---|
| YOLO | `models/yolo11n.pt`, conf=0.25, iou=0.70, imgsz=640, classes=[0], cpu |
| Tracker | `configs/tracking/bytetrack_stage3.yaml` |
| Source | `data/enrollment_clips/target_001_external_enrollment_v1.mp4` |
| Interval | frames 0–783 |

## Review package

- 1 annotated MP4 with readable `EXT_###` labels (no raw track IDs)
- 4 temporal overview PNGs (~1 s sampling)
- candidate-index PNGs (≤12/sheet)
- blank CSV template (multiple later `target_occurrence_yes` allowed)

## Status

`COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY`

Exact next gate:
`STAGE5D-B1E-C_TARGET_001_EXTERNAL_SEED_MANUAL_REVIEW_AND_FREEZE`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_tracking_seed_review_package.py \
  --config configs/reid/external_tracking_seed_review_stage5d_target_001.yaml
```
