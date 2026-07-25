# Stage 5D-B1E-F — Target Frozen Anchor OSNet Gallery Build

## Purpose

Build an immutable individual gallery for `target_001` from the seven
human-approved external anchor crops using canonical OSNet x1.0
preprocessing and the Market1501 softmax checkpoint.

This gate:

- embeds only the frozen approved anchors
- publishes individual / centroid / medoid gallery artifacts
- audits internal cosine consistency (diagnostic only)

This gate does **not**:

- run sample.mp4 target retrieval or identity assignment
- select a ReID threshold or fusion weight
- enroll automatic / pseudo-label members
- run YOLO, ByteTrack, PARSeq, or OCR
- copy crops into the gallery output root
- use the eight non-approved reviewed crops

## Approved input IDs (exact order)

1. `target_001_ext_anchor_001`
2. `target_001_ext_anchor_003`
3. `target_001_ext_anchor_004`
4. `target_001_ext_anchor_006`
5. `target_001_ext_anchor_008`
6. `target_001_ext_anchor_011`
7. `target_001_ext_anchor_014`

## Canonical runtime

- Environment: `sn-reid-cpu`
- sn-reid root: `/home/enesturkoglu2/projects/soccernet/sn-reid`
- sn-reid commit: `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`
- Adapter: `src/football_analytics/reid/embedding.py`
- Checkpoint SHA-256:
  `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`
- Preprocess: BGR→RGB, resize 256×128, ImageNet mean/std, CPU float32
- Stored / scoring embeddings: L2-normalized 512-D (`embed_tensors`)

## Run

```bash
conda run -n sn-reid-cpu \
  env PYTHONPATH=/home/enesturkoglu2/projects/football-analytics/src:/home/enesturkoglu2/projects/soccernet/sn-reid \
  python scripts/run_reid_target_gallery_build.py \
  --config configs/reid/target_gallery_build_stage5d_target_001.yaml
```

## Output root

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1`

Expected artifact budget:

- embedding NPY = 1
- individual gallery NPY = 1
- centroid NPY = 1
- medoid NPY = 1
- pairwise cosine NPY = 1
- crop copies / PNG / MP4 = 0
- sample.mp4 inference rows = 0
- identity assignments = 0

## Success statuses

- `COMPLETED_STAGE5D_B1E_F_TARGET_001_GALLERY_BUILT_READY_FOR_VALIDATION`
  → next: `STAGE5D-F1_TARGET_001_INDEPENDENT_SAMPLE_RETRIEVAL_VALIDATION_DESIGN`
- `COMPLETED_STAGE5D_B1E_F_TARGET_001_GALLERY_INTERNAL_REVIEW_REQUIRED`
  → next: `STAGE5D-B1E-F2_TARGET_001_GALLERY_INTERNAL_OUTLIER_REVIEW`

Gallery readiness is technical only: not deployment readiness, not ReID
proof, and not permission to assign identities on sample.mp4.
