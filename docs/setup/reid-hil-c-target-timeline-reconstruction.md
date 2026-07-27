# ReID-HIL-C — Verified Target Timeline Reconstruction

## Amaç
HIL-A append-only kararlarından fail-closed `target_timeline_v1` üretmek.

## Kaynak sınıflandırma
- `PRODUCT_APPROVED` — timeline eligible confirm
- `PRODUCT_UNQUALIFIED_TEST_DECISION` — örn. `dec_6fbbcc997aff`
- `ACCEPTANCE_ISOLATED` — HIL-B-R2 acceptance log
- `FIXTURE_DEMO`
- `REVOKED_OR_SUPERSEDED`
- `INVALID_PROVENANCE`

## Entrypoint
```bash
PYTHONPATH=src python scripts/run_reid_hil_c_target_timeline_reconstruction.py
```

Inference/tracking/ReID çalıştırmaz. Gap’leri interpolasyonla doldurmaz.
