# Stage 5C-C3E — PARSeq false-positive / confidence audit

Status:
**completed_discovery_set_confidence_signal_not_independently_validated**

Read-only audit of sealed Stage 5C-C3D prediction/evaluation
artifacts. No model load, inference, crop decode, or threshold
selection occurred.

## Method

- Inputs: C3D `parseq_predictions.jsonl` +
  `parseq_item_evaluation.jsonl` (+ summary/contract/manifest)
- Confidence fields used exactly as recorded:
  `sequence_confidence`,
  `confidence_method=product_of_tokenizer_decode_selected_token_probabilities`,
  `token_probabilities`, `eos_position`
- No new confidence formula
- Exhaustive descriptive operating points over unique confidence
  cuts + accept-all / accept-none sentinels
- Every operating-point row has `selected=false`
- `threshold_selected=false`

## Group counts

| Group | Count |
|---|---:|
| positive_exact | 5 |
| positive_wrong | 15 |
| positive_no_prediction | 0 |
| negative_emission | 26 |
| negative_no_emission | 0 |

## Confidence distributions (descriptive)

| Group | min | median | max |
|---|---:|---:|---:|
| positive_exact | ≈0.9934 | ≈0.9996 | ≈1.0000 |
| positive_wrong | ≈0.1990 | ≈0.7395 | ≈0.9688 |
| negative_emission | ≈0.3043 | ≈0.8874 | ≈0.9992 |

Exact vs negative confidence overlap interval is non-empty
≈ `[0.9934, 0.9992]`.

## AUROC (descriptive only; n=46, no significance claim)

| Contrast | AUROC |
|---|---:|
| exact vs negative | ≈0.938 |
| exact vs all non-exact | ≈0.961 |
| all readable-positive vs negative | ≈0.492 |

Confidence ranks exact above negatives/wrong fairly well on this
frozen set, but is **not** a strong crop-readability separator
(all-positive vs negative ≈ chance).

## Frontiers (not selected thresholds)

- Zero-negative frontier: maximum exact retained **3**, with
  wrong retained **0**, at discovery cut ≈ `0.9996361738527071`
- Perfect safe point (`exact>0`, `wrong=0`, `negative=0`) was
  observed **only on this same 46-item discovery set** at cuts
  ≈ `0.9996361738527071`, `0.9998025332713567`,
  `0.999982357095277`
- These cuts are **not** final/deployment thresholds and are **not**
  independent validation results
- All-exact retention (keep all 5 exact) still retains at least
  **4** negatives on this set

## Natural abstention

`NO_NATURAL_ABSTENTION_OBSERVED` — recognizer-only accepted a digit
on 26/26 negatives; regex rejection count 0; empty accepted output
count 0.

## Independent positive holdout

Recomputed from the frozen 78-item pilot review vs C3D POS set:

- total reviewed items = 78
- total manually readable positives = 20
- C3D readable positives = 20
- remaining independent reviewed readable positives = **0**

Recorded as:

`independent_positive_holdout_available=false`

`reason=all_existing_manually_readable_positive_items_were_used_in_c3d_discovery_set`

Therefore C3E cuts cannot validate exact retention/sensitivity on
unused positives. Remaining reviewed negatives can only support
false-positive checks.

## Legibility classifier note

A SoccerNet-finetuned legibility classifier remains a **future
candidate helper gate**. It was **not** installed or downloaded in
C3E.

## Freeze path

`outputs/reid/full_stage4b/jersey_parseq_false_positive_audit_freeze_stage5c_c3e`

## Next gate

**Stage_5C_C3F_A_independent_holdout_design**
