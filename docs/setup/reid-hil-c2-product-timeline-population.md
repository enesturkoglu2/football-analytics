# ReID HIL-C2 — Product review package and timeline approval

Bu kapı fixture/acceptance loglarından ayrı bir **product** review package üretir,
insan kararlarını `target_timeline_decision_approval_v1` ile onaylar ve HIL-C
reconstruction’ı yalnız onaylı kararlarla yeniden çalıştırır.

## Çalıştırma

```bash
export PYTHONNOUSERSITE=1
PYTHONPATH=src python scripts/run_reid_hil_c2_product_timeline_population.py
```

`--skip-ui-launch` ile UI açmadan audit/timeline snapshot alınabilir.

UI (yalnız 127.0.0.1):

```bash
python scripts/run_hil_offline_review_ui.py \
  --review-package outputs/reid/target_001_reid_hil_c2_product_review_package/review_package.json
```

1. Initial enrollment event’inde hedefi Confirm edin.
2. **Timeline Approvals** sekmesinde özeti okuyup açık checkbox ile approve edin.
3. HIL-C2 script’ini yeniden çalıştırın (veya mevcut session log’larıyla qualify edin).

## Kurallar

- Acceptance / fixture decision ID’leri approve edilemez.
- `dec_6fbbcc997aff` otomatik onaylanmaz.
- Development holdout ürün maçı gibi sunulmaz.
- Detection/tracking/ReID yeniden çalıştırılmaz.
- Approval olmadan `timeline_eligible=false`.
