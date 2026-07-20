# sn-reid general ReID checkpoint report (Stage 4A-4)

- **Date:** 2026-07-21
- **sn-reid commit:** `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`
- **Environment:** `sn-reid-cpu` (unchanged; no pip/conda updates this stage)

## Selected checkpoint

| Field | Value |
|---|---|
| Documented model name | `osnet_x1_0` |
| Source document | `sn-reid/docs/MODEL_ZOO.md` — section **Same-domain ReID** |
| Link type | Official Google Drive file link in MODEL_ZOO |
| Google Drive file ID | `1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA` |
| URL | `https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing` |
| Architecture | OSNet `osnet_x1_0` (~2.2M params, 0.98 GFLOPs in MODEL_ZOO) |
| Training dataset | **Market1501** (general person ReID) |
| Training method (as documented) | Loss `softmax`; input `(256, 128)`; transforms `random_flip`; distance `euclidean` |
| Named config YAML | **Not named** in the Same-domain ReID table of MODEL_ZOO |
| Reported Market1501 metrics | Rank-1 **94.2**, mAP **82.6** |
| Declared file size in MODEL_ZOO | **Not stated** |
| License / usage note in MODEL_ZOO | **None** beyond repo Torchreid/MIT context |

### Type confirmation

- **Not ImageNet-only:** ImageNet `osnet_x1_0` in MODEL_ZOO uses a different Drive ID (`1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY`). Selected ID is under Same-domain ReID / market1501 column.
- **Not SoccerNet-trained:** No SoccerNet dataset or SoccerNet-trained weight was used or claimed.
- **General human ReID checkpoint:** Market1501 same-domain person re-identification weights.

## Local file

| Field | Value |
|---|---|
| Path | `/home/enesturkoglu2/projects/soccernet/checkpoints/general-reid/osnet_x1_0_market1501_softmax_256x128.pth.tar` |
| Size | 10 399 605 bytes (~10M / 10.4M download) |
| SHA-256 | `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154` |
| `file(1)` | `data` |

## Checkpoint structure

- Top-level type: `OrderedDict` (raw weight map; **no** wrapper keys such as `state_dict` / `epoch` / `rank1` / `mAP`)
- Keys are OSNet layer names (including `fc.*` and Market1501 `classifier.*` with 751 classes)

## Controlled load smoke (`pretrained=False`)

- Method: `build_model(..., pretrained=False)` then `load_pretrained_weights(local_path)`
- **FeatureExtractor was not constructed**
- **No ImageNet / gdown auto-download** during load
- Model class: `OSNet`
- Parameters before/after: **2 170 021** / **2 170 021**
- `model.training`: `False`
- CUDA available: `False`

### Weight matching

| Outcome | Count |
|---|---|
| Matched layers | **565** |
| Discarded layers | **2** |

Discarded (expected size mismatch with `num_classes=1` vs Market1501 751-ID classifier):

- `classifier.weight` checkpoint `(751, 512)` vs model `(1, 512)`
- `classifier.bias` checkpoint `(751,)` vs model `(1,)`

No non-classifier mismatches. Backbone + `fc` loaded successfully. Repo code was not modified.

## What was not done

- No SoccerNet dataset download
- No ImageNet pretrained weight download
- No second checkpoint / other architectures
- No FeatureExtractor, training, evaluation, or crop embedding / forward pass
- No package install/update; `football-cv` untouched
- No git commit/push
