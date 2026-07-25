# Stage 5D-F3 — Independent Retrieval Scoring and Evaluation

## Purpose

Score the 150 existing sample.mp4 scoreable embeddings against the
frozen target_001 gallery-v1 (7 anchors). Compute threshold-free
segment and component retrieval metrics under the F2A ground truth
and F2B amended strong-signal contract.

## Allowed

- Read frozen gallery NPY and existing sample embeddings
- Cosine similarity, ranking, Recall/AP/MRR/AUROC/AUPRC

## Forbidden

- Threshold selection, identity assignment, gallery mutation
- New OSNet embeddings, YOLO/ByteTrack/OCR, sample.mp4 decode
- Changing formulas / GT / gallery after seeing scores

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_target_independent_retrieval_evaluation.py \
  --config configs/reid/target_independent_retrieval_evaluation_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F3_TARGET_001_INDEPENDENT_RETRIEVAL_EVALUATED`

Conditional next gate depends on descriptive outcome (F4 / F3A / F2C).
