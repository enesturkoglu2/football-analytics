# Stage 5C-C3D — SoccerNet-finetuned PARSeq offline smoke results

Status: **completed_exact_signal_with_negative_emission_risk**

This document freezes descriptive results from the offline
recognizer-only PARSeq smoke on the frozen Stage 5A 46-crop set.
It is **not** a deployment accuracy claim and selects **no**
confidence threshold.

## Provenance (C3B / C3C)

- Isolated environment: `sn-jersey-parseq-cpu` (Python 3.9.25,
  torch `1.13.1+cpu`, torchvision `0.14.1+cpu`,
  pytorch-lightning `1.9.5`, timm `0.9.5`)
- Bundled code:
  `/home/enesturkoglu2/projects/external/jersey-number-pipeline/str/parseq`
  at external HEAD `007d54e5530a66616ed5081ca35e0028b36aadb5`
- Checkpoint (Git-ignored):
  `.../jersey-parseq/soccernet-finetuned/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt`
- Local SHA-256:
  `14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879`
- Byte size: `381608677`
- Official full SHA-256: unavailable
- Generic bundled `parseq-bb5792a6.pt` was **not** loaded

## Offline CPU runtime contract

- Device: CPU only (`cuda_available=false`)
- Network policy on smoke: `pass_loopback_only`
- Model class: `PARSeq`
- Parameter count: `23832671`
- Dtype: float32
- Missing / unexpected keys: `[]` / `[]`
- Jersey decode overlay (static `str.py` reference):
  `logits[:, :3, :11]` then bundled tokenizer decode
- Pickled hparams were disclosed (for example
  `max_label_length=25`); they were not rewritten to fake a digits-only
  contract

## Frozen 46-item input

Byte-identical C1 freeze manifests:

- inference:
  `f6d8ffa9c8b1c00d8861a73337288e4de9ebf0610f947f3b1d920f94dc2abf39`
- evaluation reference:
  `fd1684606e7ced670f8fb8938fbadd8a20afb686d14d2e6b2bee3d746d512022`

Class mix: POS=20, A=10, B=2, C=7, D=2, E=5.

## Results (descriptive)

| Set | Metric | Count |
|---|---|---:|
| Positive 20 | exact | **5** |
| Positive 20 | wrong | **15** |
| Positive 20 | no-prediction | **0** |
| Negative 26 | accepted digit emission | **26** |
| All 46 | inference_error | **0** |

Confidence method recorded in predictions:

`product_of_tokenizer_decode_selected_token_probabilities`

Approximate per-item runtime median ≈ 153 ms; inference wall ≈ 7.0 s;
peak RSS ≈ 749352 KB.

## Evidence labels

- `PARSEQ_EXACT_SIGNAL_PRESENT`
- `PARSEQ_NEGATIVE_EMISSION_RISK_PRESENT`
- `PARSEQ_CHECKPOINT_RUNTIME_CONTRACT_VALIDATED`
- `CURRENT_STAGE5A_ROI_HAS_SOME_PARSEQ_COMPATIBILITY_SIGNAL`

C3D decision at smoke time:
`GO_STAGE5C_C3E_PARSEQ_FALSE_POSITIVE_AUDIT`

## Interpretation limits

- Frozen 46-item discovery set only
- Exact signal does **not** imply usable recognizer-only deployment
- Recognizer-only path emitted digits on **all** negatives
- No confidence threshold was selected in C3D
- Jersey number is evidence, not physical identity

## Freeze path

`outputs/reid/full_stage4b/jersey_parseq_smoke_freeze_stage5c_c3d`

Source live output (immutable reference for freeze copies):

`outputs/reid/full_stage4b/jersey_parseq_smoke_stage5c_c3d`
