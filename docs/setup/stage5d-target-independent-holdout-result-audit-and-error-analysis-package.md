# Stage 5D-F3N — Independent Holdout Result Audit and Error Analysis Package

## Purpose

Build a visual, read-only error-analysis package over the frozen F3M holdout v2 result:

1. Validate F3M score/rank/metric/seal artifacts without recomputation.
2. Validate F3L GT, F3K crop lineage, and F3G gallery crop lineage.
3. Materialize exact 50 audit items (10 positives + 20 same-team FP + 20 other-team FP).
4. Render query–target–distractor comparison contact sheets and diagnostic videos.
5. Emit a blank human root-cause template (no automatic diagnoses).
6. Record holdout retirement and deferred cleanup governance (no deletions).

## Commands

```bash
cd /home/enesturkoglu2/projects/football-analytics

conda run -n football-cv python \
  scripts/run_reid_independent_holdout_v2_result_audit_and_error_analysis.py \
  --config configs/reid/independent_holdout_v2_result_audit_stage5d_target_001.yaml

conda run -n football-cv python -m unittest discover -s tests \
  -p 'test_reid_independent_holdout_v2_result_audit_and_error_analysis.py' -v
```

## Hard constraints

- No score/rank/metric recomputation
- No OSNet embeddings
- No gallery mutation / enrollment / hard-negative mining
- No automatic root-cause or performance-class invention
- No upstream deletion or cleanup
- Holdout MP4 decode only for diagnostic render

## Expected package

- Audit items: 50
- Contact sheets: 5
- Diagnostic videos: 3
- Manual root-cause decisions: 0

## Next gate

`STAGE5D-F3O_TARGET_001_INDEPENDENT_HOLDOUT_MANUAL_ERROR_INTERPRETATION_AND_BASELINE_CLOSURE`
