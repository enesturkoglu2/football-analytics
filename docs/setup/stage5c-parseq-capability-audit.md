# Stage 5C-C3A — SoccerNet fine-tuned PARSeq capability audit

- **Status:** `completed_audit_no_install`
- **Stop/go:** `GO_STAGE5C_C3B_ENV_PLAN`
- **Date context:** Stage 5C-C2 closed with no exact signal; next
  candidate family is SoccerNet-finetuned PARSeq
- **This gate did not:** create a PARSeq environment, install packages,
  download the fine-tuned checkpoint, load weights, or run inference

## External repository

| Field | Value |
|---|---|
| Path | `/home/enesturkoglu2/projects/external/jersey-number-pipeline` |
| Origin | `https://github.com/mkoshkina/jersey-number-pipeline.git` |
| Branch | `main` |
| HEAD | `007d54e5530a66616ed5081ca35e0028b36aadb5` |
| Commit date | 2024-10-07T12:09:56-04:00 |
| Working tree | clean |
| Paper | CVPRW 2024 CVsports — Koshkina & Elder |

Tracked binary present in the clone (not SoccerNet fine-tuned):

- `str/parseq/models/parseq-bb5792a6.pt` (~91 MB) — original PARSeq
  pretrained weight shipped inside the upstream tree

## Full pipeline vs minimal recognizer

Full pipeline components (from README / `main.py` / `setup.py`):

- SAM
- Centroid-ReID
- ViTPose (pose-guided torso crops)
- legibility classifier
- PARSeq recognizer
- tracklet consolidation (`-1` for illegible at aggregation)

Minimal recognizer-only path (import graph of `str.py`):

- bundled `str/parseq` + `load_from_checkpoint`
- PIL RGB transform / tokenizer decode
- **does not** import SAM, ReID, ViTPose, or legibility helpers

`setup.py` is **unsafe for the current gate**
(`setup.py_safe_for_current_gate=false`): creates conda envs, clones
repos, downloads weights (`gdown` / `urlretrieve`), and assumes CUDA
wheels. Minimal smoke must bypass `setup.py`.

## Bundled PARSeq provenance

- Classification: **`bundled_code_present_commit_unresolved`**
- PARSeq code is present under `str/parseq/` (not an empty placeholder;
  no `.gitmodules`)
- Exact upstream commit/tag is **not** recorded in-repo
- Current `baudm/parseq` `main` tip differs; do **not** assume a fresh
  upstream clone is drop-in compatible without an adapter
  (`incompatible_with_current_upstream_without_adapter` risk)
- Recommendation for smoke: use the **bundled** `str/parseq` tree

## SoccerNet fine-tuned checkpoint (metadata only)

| Field | Value |
|---|---|
| Source | `configuration.py` / README |
| Google Drive file ID | `1uRln22tlhneVt3P6MePmVxBWSLMsL3bm` |
| Expected local filename | `models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt` |
| Format | Lightning `.ckpt` |
| Metadata size | ~364M (from Drive virus-scan HTML; exact byte size not from Content-Length) |
| Access class | **`publicly_accessible_metadata_resolved`** (confirmation page; not login-walled) |
| Official SHA-256 | **`official_full_checksum_available=false`** |

The checkpoint **body was not downloaded** in C3A.

Legibility weights exist separately (~81M) and are **not required** for
initial recognizer-only smoke.

## Checkpoint–model / input contract (static)

Verified statically from bundled configs / `str.py`:

- Input: PIL **RGB** → BICUBIC resize → `Normalize(0.5, 0.5)`
- Default model `img_size`: **[32, 128]** (H×W)
- `max_label_length=2`
- Digit charset for jersey decode (`charset_test` / `string.digits`)
- Inference decode slice in `str.py`: `logits[:, :3, :11]` then
  tokenizer decode
- Exact special-token layout vs checkpoint hparams:
  **`requires_runtime_validation`**

Official SoccerNet path uses pose-guided torso crops. Compatibility of
our Stage 5A number-search ROI is **not** proven:
`roi_compatibility_requires_smoke`.

## CPU / dataset / license

- CPU classification: **`cpu_supported_with_small_adapter`**
  (`--device` exists; default is `cuda`; temperature-calibration path
  hard-codes CUDA but is not required for inference)
- Bundled `requirements/core.txt` already lists CPU wheels
  (`torch==1.13.1+cpu`, `torchvision==0.14.1+cpu`,
  `pytorch-lightning==1.9.5`, `timm==0.9.5`) — pins to be locked in C3B
- Dataset: **`dataset_not_required_for_initial_local_smoke`**
- Pipeline license: Creative Commons **BY-NC 3.0**
- Bundled PARSeq source: Apache-2.0
- Fine-tuned checkpoint: separate license not verified
  (`checkpoint_license_verified=false`; local research smoke only;
  `requires_external_legal_review=true` for redistribution/commercial)

## Recommended next environment (not created)

- Proposed name: **`sn-jersey-parseq-cpu`**
- Must remain isolated from `football-cv`, `sn-reid-cpu`, and
  `sn-jersey-mmocr-cpu`
- No install/download performed in this audit gate

## Integration recommendation

Primary: clean local / isolated recognizer-only adapter over bundled
PARSeq + SoccerNet fine-tuned `.ckpt` + Stage 5A ROI (BGR→RGB in
adapter).

Full SAM / ViTPose / ReID / legibility pipeline:
**`not_recommended_for_initial_smoke`**.

## Next gate

**Stage 5C-C3B** — isolated PARSeq CPU environment plan and controlled
checkpoint acquisition policy (still no automatic install in this
document). Later: C3C controlled asset acquisition, C3D offline
recognizer-only smoke on the frozen 46-crop set.
