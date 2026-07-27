# ReID HIL-A — Event model and append-only decision schema

## Scope
Implements recovery event validation, candidate manifest validation, human
decision invariants, and append-only JSONL decision logs for the HIL MVP.

## Modules
- `football_analytics.reid.hil.events`
- `football_analytics.reid.hil.candidates`
- `football_analytics.reid.hil.decisions`
- `football_analytics.reid.hil.log`
- `football_analytics.reid.hil.resolve`

## CLI
```bash
PYTHONPATH=src python scripts/run_reid_hil_a_event_decision_schema.py --demo-out /tmp/hil_a_demo
PYTHONPATH=src python scripts/run_reid_hil_a_event_decision_schema.py --validate-log /tmp/hil_a_demo/demo_decision_log.jsonl
```

## Non-goals
No Streamlit/UI, no ReID inference/scoring, no CLIP download, no package installs.
