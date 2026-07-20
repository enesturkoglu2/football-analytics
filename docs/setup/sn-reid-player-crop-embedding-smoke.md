# sn-reid player-crop embedding smoke (Stage 4A-5)

- **Date:** 2026-07-21
- **Purpose:** Single real-crop embedding smoke on CPU. Not identity accuracy evaluation.
- **sn-reid commit:** `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`
- **Environment:** `sn-reid-cpu` (no package changes this stage)

## Selected observation

| Field | Value |
|---|---|
| `track_id` | 463 |
| Valid observations | 371 (all in-bounds, positive size, with confidence) |
| Selection rule | Largest bbox area → higher confidence → smaller `frame_index` |
| `frame_index` | **409** |
| `bbox_xyxy` | `[231.0178680419922, 389.2264404296875, 288.88677978515625, 502.9831848144531]` |
| `detection_confidence` | 0.7585912346839905 |
| Bbox w×h / area | 57.87 × 113.76 / 6582.98 |

## Crop

| Field | Value |
|---|---|
| Path | `outputs/reid/smoke/player_crop.jpg` |
| Original crop size | **58 × 114** (W×H) |
| JPEG bytes | 3750 |
| Overlay / resize on disk | None (raw detection crop only) |

## Preprocessing (manual; no FeatureExtractor)

1. OpenCV BGR → RGB  
2. PIL Image  
3. `Resize((256, 128))` (H×W)  
4. `ToTensor`  
5. ImageNet normalize `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`  
6. Batch shape `(1, 3, 256, 128)`, float32, finite  

## Model and checkpoint

| Field | Value |
|---|---|
| Model | `osnet_x1_0` via `build_model(..., pretrained=False, use_gpu=False)` |
| Load | `load_pretrained_weights(local_path)` |
| Checkpoint | Market1501 **general person ReID** (not SoccerNet-trained) |
| Path | `/home/enesturkoglu2/projects/soccernet/checkpoints/general-reid/osnet_x1_0_market1501_softmax_256x128.pth.tar` |
| SHA-256 | `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154` |
| Discarded keys | `classifier.weight` / `classifier.bias` (751→1; expected) |
| FeatureExtractor | **Not used** |
| `pretrained` | **False** |
| Automatic ImageNet/gdown/dataset download | **None** |

## Embedding

| Field | Value |
|---|---|
| Path | `outputs/reid/smoke/embedding.npy` |
| Shape / dtype | `(512,)`, `float32` |
| Raw L2 norm | ~24.347 |
| Normalized L2 norm | ~1.0 |
| Device | CPU |

## Repeatability smoke (same tensor, same model, twice)

| Metric | Value |
|---|---|
| Max abs diff (raw) | **0.0** |
| Cosine similarity (normalized) | **~1.0** |

This only checks deterministic forward reproducibility. Same/different player discrimination was **not** measured.

## Outputs

`outputs/reid/smoke/` contains only:

- `player_crop.jpg`
- `embedding.npy`
- `embedding_metadata.json` (`status=ok`)

## Stage note

This is a controlled **smoke** of crop → embedding. It does **not** close product ReID / track linking (Stage 4B).
