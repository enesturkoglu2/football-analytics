# sn-reid-cpu install report (Stage 4A-3)

- **Date:** 2026-07-21T01:28:08+03:00
- **sn-reid commit:** `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`
- **SoccerNet commit:** `74461027ac2095ce2f8d4ee991eccb5dd5f42459`
- **Environment name:** `sn-reid-cpu`
- **Environment path:** `/home/enesturkoglu2/miniconda3/envs/sn-reid-cpu`
- **Disk size (after install):** 1.7G

## Versions

| Component | Version |
|---|---|
| Python | 3.10.20 |
| torch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| NumPy | 2.2.6 |
| OpenCV (`opencv-python-headless`) | 5.0.0 / package 5.0.0.93 |
| SoccerNet (PyPI) | 0.1.62 |
| tensorboard | 2.21.0 |

## CPU-only verification

- `torch.cuda.is_available()` → `False`
- `torch.version.cuda` → `None`
- Torch/vision installed from `https://download.pytorch.org/whl/cpu` (`+cpu` wheels)

## Install approach

- Created isolated conda env with `python=3.10` + `pip`
- Did **not** install sn-reid `requirements.txt` as a whole
- Did **not** run `pip install -e .`, `setup.py`, or Cython extension builds
- Minimum explicit packages: `numpy`, `Pillow`, `opencv-python-headless`, `scipy`, `tensorboard`, `SoccerNet`, `gdown`, `six`
- SoccerNet pulls transitive deps (e.g. matplotlib, boto3, huggingface_hub, pycocoevalcap); these are package metadata deps, not an intentional full training stack
- Only `opencv-python-headless` (not `opencv-python`)
- `football-cv` left untouched

## PYTHONPATH approach

- sn-reid used read-only via:
  `PYTHONPATH=/home/enesturkoglu2/projects/soccernet/sn-reid`
- No editable install; no permanent shell profile change

## Smoke test (`pretrained=False`)

- `import torchreid` → OK (Cython rank warning expected; Python fallback)
- `build_model('osnet_x1_0', num_classes=1, pretrained=False, use_gpu=False)` → OK
- Model class: `OSNet`
- Parameter count: `2170021`
- `model.eval()` → `training mode: False`
- **FeatureExtractor was not constructed**
- **No `load_pretrained_weights`**
- **No random-tensor forward / real inference**
- **No checkpoint or dataset download**

## Artifacts

- `docs/setup/sn-reid-cpu-after-install.txt` — `pip freeze`
- `docs/setup/sn-reid-cpu-after-install.yml` — `conda env export --no-builds`
