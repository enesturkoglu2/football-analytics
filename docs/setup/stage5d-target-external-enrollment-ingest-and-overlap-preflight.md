# Stage 5D-B1E-A — External Enrollment Clip Ingest and Overlap Preflight

## Purpose

Ingest the immutable external enrollment clip for `target_001`, prove it is
not an exact/file duplicate of evaluation `sample.mp4`, audit visual overlap
with deterministic perceptual fingerprints (not model inference), and publish
eligible non-overlap intervals for a later seed-review gate.

This gate does **not** run detection, tracking, embedding, OCR, similarity, or
target selection.

## Sources

| Role | Path | SHA-256 |
|---|---|---|
| Enrollment | `data/enrollment_clips/target_001_external_enrollment_v1.mp4` | `ab1c622c…73e877` |
| Evaluation | `data/test_clips/sample.mp4` | `f4b28dd5…bfd7b9b` |

External source is enrollment-only and must not be used for evaluation.
`sample.mp4` remains evaluation-only.

## Overlap audit

- Coarse sample every 15 frames → dHash candidates
- Refine around candidates (stride 3)
- Verify with normalized correlation + MAE on resized grayscale
- Merge contiguous verified pairs
- Eligible intervals require ≥5 s continuous non-overlap

## Statuses

| Decision | Final status | Next gate |
|---|---|---|
| Non-overlapping + eligible | `COMPLETED_STAGE5D_B1E_A_EXTERNAL_ENROLLMENT_PREFLIGHT_READY` | `STAGE5D-B1E-B_…DETECTION_TRACKING_AND_SEED_REVIEW_PACKAGE` |
| Partial overlap + eligible | `COMPLETED_STAGE5D_B1E_A_PARTIAL_OVERLAP_ELIGIBLE_INTERVALS_READY` | `STAGE5D-B1E-B_…ELIGIBLE_INTERVAL_PROCESSING` |
| Ineligible overlap | `COMPLETED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INELIGIBLE_OVERLAP` | `STAGE5D-B1E_…REPLACEMENT_EXTERNAL_CLIP_REQUIRED` |

## Artifacts

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_enrollment_preflight`

- source / overlap / eligibility audits
- one overview PNG (no boxes / OCR / IDs)
- blank external seed-review template JSON

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_enrollment_preflight.py \
  --config configs/reid/external_enrollment_preflight_stage5d_target_001.yaml
```
