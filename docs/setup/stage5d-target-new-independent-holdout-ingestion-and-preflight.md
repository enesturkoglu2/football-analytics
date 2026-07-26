# Stage 5D-F3I — New Independent Holdout Ingestion and Preflight

## Purpose

Ingest `data/test_clips/target_001_independent_holdout_v2.mp4` and prove it is
technically valid and independent from `sample.mp4` and external enrollment via
deterministic frame fingerprints.

No detection, tracking, crops, embeddings, scoring, threshold, or identity.

## Status / readiness

- `COMPLETED_STAGE5D_F3I_TARGET_001_NEW_INDEPENDENT_HOLDOUT_INGESTED_AND_PREFLIGHT_PASSED`
- `TARGET_001_INDEPENDENT_HOLDOUT_V2_READY_FOR_LABEL_BLIND_UNIVERSE_BUILD`

Exact next gate:
`STAGE5D-F3J_TARGET_001_NEW_INDEPENDENT_HOLDOUT_LABEL_BLIND_DETECTION_TRACKING_AND_SEGMENT_UNIVERSE_BUILD`

## Run

```bash
conda run -n football-cv python scripts/run_reid_independent_holdout_v2_ingestion_and_preflight.py \
  --config configs/reid/independent_holdout_v2_ingestion_stage5d_target_001.yaml
```
