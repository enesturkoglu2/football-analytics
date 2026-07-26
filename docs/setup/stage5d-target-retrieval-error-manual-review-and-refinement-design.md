# Stage 5D-F3B — Retrieval Error Manual Review and Refinement Design

## Purpose

Freeze human diagnostic findings from F3A and design an external-only
refinement protocol. No gallery mutation, scoring, embeddings,
threshold, or identity assignment.

## Key frozen conclusions

- False positives are predominantly same-uniform confusion
- Four conflicting components are grouping-overmerge
- Anchor roles classified; all `removal_authorized=false`
- Sample.mp4 is analysis-only; new independent holdout required

## Run

```bash
conda run -n football-cv \
  python scripts/run_reid_retrieval_error_manual_review_and_refinement_design.py \
  --config configs/reid/retrieval_error_manual_review_and_refinement_design_stage5d_target_001.yaml
```

## Success

`COMPLETED_STAGE5D_F3B_TARGET_001_REFINEMENT_DESIGN_READY`

Exact next gate:
`STAGE5D-F3C_TARGET_001_EXTERNAL_ONLY_HARD_NEGATIVE_AND_VIEW_ANCHOR_REVIEW_PACKAGE`
