# ReID-R2B — Market1501 Multi-Frame Tracklet Ablation (Completion)

## Purpose

Measure whether quality-aware multi-frame tracklet query representation improves
Target 001 holdout-v2 retrieval **without** changing the Market1501 OSNet
checkpoint, frozen gallery-v2 / distractor-v1, or frozen `S = T_max − D_max`
scoring formula.

Holdout v2 role for this gate: `development_and_error_analysis_only`.

## Final outcome

`COMPLETED_R2B_MULTIFRAME_REGRESSION` (`MULTIFRAME_REGRESSION`)

Selected development candidate among B variants: `B3` (least-bad AP among B1/B2/B3;
still worse than frozen Experiment A).

This is **not** a deployment claim and **not** an independent final-test result.

## Tracked entrypoints

| Asset | Path |
|-------|------|
| Contract (pre-registered variants/outcomes) | `configs/reid/r2b_market1501_multiframe_contract.yaml` |
| Helpers | `src/football_analytics/reid/multiframe_r2b.py` |
| Runner | `scripts/run_reid_r2b_market1501_multiframe.py` |
| Tests | `tests/test_reid_r2b_multiframe.py` |

## Run

```bash
cd /home/enesturkoglu2/projects/football-analytics

PYTHONPATH="src:/home/enesturkoglu2/projects/soccernet/sn-reid" \
  /home/enesturkoglu2/miniconda3/envs/sn-reid-cpu/bin/python \
  scripts/run_reid_r2b_market1501_multiframe.py

PYTHONPATH=src /home/enesturkoglu2/miniconda3/envs/football-cv/bin/python \
  -m unittest discover -s tests -p 'test_reid_r2b_multiframe.py' -q
```

## Artifact root (gitignored / not committed)

`outputs/reid/target_001_reid_r2b_market1501_multiframe`

## Exact next gate

`REID_R2C_FOOTBALL_DOMAIN_CHECKPOINT_ACQUISITION_AND_SMOKE`
