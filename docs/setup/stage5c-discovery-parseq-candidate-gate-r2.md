# Stage 5C — Discovery-primary PARSeq candidate gate (rebuild r2)

- **Gate freeze:** REBUILD-R6A
- **Inference / derivation gate:** REBUILD-R6
- **Canonical split generation:** `r2_capacity_balanced`
- **Exact next gate:**
  `REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE`

**This result does not prove general model accuracy.**  
The derived cut is an **independent holdout-validation candidate
only**. It is **not** a deployment threshold and **not** a calibrated
probability.

## 1. Provenance roots

| Role | Path |
|---|---|
| Canonical split | `outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced` |
| Discovery-primary annotation freeze | `outputs/reid/full_stage4b_rebuild_r2_stage5c_discovery_primary_annotation_freeze` |
| Discovery PARSeq gate output | `outputs/reid/full_stage4b_rebuild_r2_stage5c_discovery_primary_parseq_gate` |

Annotation freeze (REBUILD-R5):

- 40 discovery-primary rows reviewed (`annotation_method=human_visual_review_assistant_supported`)
- final_approver / reviewer = Furkan
- target-model / PARSeq / OCR predictions unseen at annotation time
- historical 78-pilot labels unused
- discovery reserve **not** opened
- holdout **not** opened

## 2. Discovery-primary contract

- universe join: discovery_primary manifest ↔ frozen annotations
- rows = **40**
- classes:
  - readable_positive = **10**
  - non_readable_negative = **27**
  - uncertain_excluded = **3** (diagnostic only; not in gate safety math)

Discovery annotation minima were already met in R5:

- readable positive ≥ 8 → 10
- non-readable/negative ≥ 20 → 27
- therefore discovery reserve remained closed for performance/
  threshold search

## 3. Checkpoint / repository

- Checkpoint:
  `…/jersey-parseq/soccernet-finetuned/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt`
- Bytes: 381608677
- SHA-256:
  `14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879`
- External repo:
  `/home/enesturkoglu2/projects/external/jersey-number-pipeline`
- Commit: `007d54e5530a66616ed5081ca35e0028b36aadb5`
- Environment: `sn-jersey-parseq-cpu` (offline CPU; no download)

## 4. Frozen preprocessing / confidence

Frozen **before** discovery predictions
(`runtime/preprocessing_contract_pre_inference.json`):

- input region: clean-universe canonical number ROI
- ROI padding/expansion: 0
- resize: 32×128 (H×W), BICUBIC
- color: BGR crop slice → RGB transform
- normalize mean/std: 0.5 / 0.5
- decode: `str.py` logits slice 3×11 + tokenizer decode
- confidence metric:
  `product_of_tokenizer_decode_selected_token_probabilities`
- top-1 only; no stochastic augmentation
- prediction normalize: surrounding whitespace only; leading zeros
  preserved (`09` ≠ `9`)

Historical C3E confidence cut is **forbidden** as fallback.

## 5. Determinism

Two-pass discovery inference under the same frozen contract:

- normalized prediction exact match: 40/40
- valid/invalid flag exact match: 40/40
- confidence absolute difference ≤ 1e-12: 40/40

## 6. Discovery outcomes (item-level)

Positive / readable=yes (10):

- exact = **5**
- wrong = **5**
- no_prediction = **0**

Negative / readable=no (27):

- negative_digit_emission = **27**
- negative_no_prediction = **0**

Uncertain (3; diagnostic):

- uncertain_digit_emission = **3**
- uncertain_no_prediction = **0**

**Recognizer-only negative-emission risk continues:** every
non-readable-negative item emitted a valid 1–2 digit string under the
frozen recognizer-only ROI contract. Safety therefore depends on the
confidence cut, not on “no emission”.

## 7. Candidate-gate derivation

Operating points:

- unique valid-emission cuts + no-acceptance sentinel → **41**
- zero-error eligible points (`wrong=0`, `negative=0`, `exact>0`,
  non-sentinel) → **1**

Selected candidate:

| Field | Value |
|---|---|
| gate_status | `DISCOVERY_SAFE_CANDIDATE_GATE_DERIVED` |
| confidence_cut exact decimal | `0.99992299168434329` |
| confidence_cut float64 hex | `3fefff5e8079b000` |
| comparison operator | `>=` |
| accepted_exact | 1 |
| accepted_wrong_positive | 0 |
| accepted_negative | 0 |
| accepted_uncertain | 0 |
| support | **1** |
| accepted item | `discovery_primary_028` |

Selection rule (preregistered): lowest numeric cut among zero-error
points, then higher accepted_exact, then ascending cut / op id.

## 8. Interpretation limits (mandatory)

Bu sonuç modelin genel doğruluğunu kanıtlamaz.
Gate yalnız bağımsız holdout validation candidate'tır.

- **Not** a deployment threshold
  (`deployment_threshold_selected=false`)
- **Not** a calibrated probability
- **Not** a permanent accuracy claim
- Support is **low** (support=1)
- Gate is **immutable after freeze** with respect to later holdout
  outcomes (`mutable_after_holdout=false`)
- Historical C3E cut was **not** reused
- Discovery reserve stayed **closed**
- Holdout remained **unopened / unreviewed** at derivation time

## 9. Preregistered holdout decision taxonomy

When holdout primary is later annotated and evaluated against this
frozen candidate, the immutable decision classes remain:

- `PASS`
- `INCONCLUSIVE`
- `FAIL`
- `BLOCKED`

(Exact holdout minima and safety rules are those preregistered in the
clean-split / discovery-gate contracts; they are not revised by
discovery metrics.)

## 10. Canonical active code

- `scripts/run_reid_jersey_parseq_discovery_gate.py`
- `configs/reid/jersey_parseq_discovery_gate_stage5c_rebuild_r2.yaml`
- `tests/test_reid_jersey_parseq_discovery_gate.py`

Old C3F-A holdout triad is preserved for history/tests only and must
not drive the r2 discovery/holdout protocol.

## 11. Exact next gate

`REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE`

Holdout may open only because a candidate gate was successfully
frozen; the cut itself must not be retuned after holdout results.
