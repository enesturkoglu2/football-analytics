# Stage 5C holdout PARSeq validation and Stage 5C closure (rebuild r2)

- **Gate:** REBUILD-R8 + REBUILD-R8A
- **Holdout validation decision:** `INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT`
- **Stage 5C status:** `closed`
- **Exact next gate:**
  `STAGE5D-A_TARGET_GALLERY_ENROLLMENT_DESIGN_AND_PREFLIGHT`

## 1. Provenance

| Artifact | Path / value |
|---|---|
| Canonical split | `…/stage5c_clean_split_capacity_balanced` (`r2_capacity_balanced`) |
| Discovery annotation freeze | `…/discovery_primary_annotation_freeze` |
| Discovery candidate gate | `…/discovery_primary_parseq_gate` |
| Holdout annotation freeze | `…/holdout_primary_annotation_freeze` |
| Holdout validation root | `…/holdout_primary_parseq_validation` |
| Stage 5C closure root | `…/stage5c_closure` |
| Frozen discovery cut | `0.99992299168434329` (`>=`, hex `3fefff5e8079b000`) |
| Checkpoint SHA | `14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879` |
| Preprocessing | clean-universe ROI, pad=0, 32×128 BICUBIC, product-of-token-probs |

## 2. Holdout annotation contract

- 48 holdout-primary items
- readable positive / non-readable negative / uncertain = **16 / 30 / 2**
- sufficiency: `HOLDOUT_PRIMARY_ANNOTATION_SUFFICIENT`
- holdout reserve closed

## 3. Independent holdout PARSeq results (R8)

Two-pass determinism: prediction/valid/confidence **48/48**.

### A. Raw recognizer outcomes

| Class | Counts |
|---|---|
| Positive exact / wrong / no_prediction | **5 / 11 / 0** |
| Negative digit emission / no_prediction | **30 / 0** |
| Uncertain digit emission / no_prediction | **2 / 0** |

Recognizer alone does **not** abstain: negatives emit digits **30/30**.

### B. Frozen candidate gate outcomes

| Metric | Value |
|---|---|
| accepted total / exact / wrong / negative / uncertain | **0 / 0 / 0 / 0 / 0** |
| rejected exact | 5 |
| max exact confidence | `0.999913811844408` (< cut) |

False acceptance observed = false.  
True acceptance observed = false.  
Independent utility demonstrated = false.

**Doğru ifade:** Frozen gate bağımsız holdout’ta yanlış kabul üretmedi;
ancak doğru kabul de üretmediği için pratik utility doğrulanamadı.

Bu sonuç:

- gate PASS iddiası değildir
- jersey OCR’nin genel güvenilirliğini kanıtlamaz
- deployment-ready / calibrated probability değildir
- “model tamamen işe yaramaz” iddiası da değildir

## 4. Threshold / reserve policy

- discovery candidate cut **unchanged**
- no holdout threshold retune / optimization / retraining
- holdout reserve remained **closed**
- current holdout **not** reusable for threshold selection or as a
  fresh validation set after reopening

## 5. Stage 5C closure policy

Automated PARSeq jersey evidence for Stage 5E:

- mode = **`diagnostic_only`**
- may not enter fusion score
- may not create identity
- may not veto appearance matches
- may not enroll gallery identities

Appearance ReID remains primary. Unknown identity is preserved.
Stage 5D is **not** blocked.

Future jersey reopening requires:

- new preregistration
- new discovery data
- new independent holdout

## 6. Canonical code

Active holdout-validation triad:

- `scripts/run_reid_jersey_parseq_holdout_validation.py`
- `configs/reid/jersey_parseq_holdout_validation_stage5c_rebuild_r2.yaml`
- `tests/test_reid_jersey_parseq_holdout_validation.py`

Discovery-gate triad remains canonical for provenance.  
Old C3F-A triad remains preserved/superseded (historical/tests only).

## 7. Exact next gate

`STAGE5D-A_TARGET_GALLERY_ENROLLMENT_DESIGN_AND_PREFLIGHT`

Stage 5D-A will design appearance-driven target gallery enrollment
with automated PARSeq jersey evidence disabled for identity use.
