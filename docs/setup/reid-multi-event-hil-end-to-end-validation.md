# Multi-event HIL end-to-end validation

İzole match-specific product review: enrollment → gallery crop onayları →
SportsReID gallery embed (helper) → recovery HIL → timeline approval → overlay.

## Çalıştırma

```bash
export PYTHONNOUSERSITE=1
PYTHONPATH=src python scripts/run_reid_multi_event_hil_end_to_end_validation.py
```

UI yalnızca 127.0.0.1. Eski development gallery kullanılmaz.

Overlay izledikten sonra:

`outputs/reid/target_001_multi_event_hil_review_package/session/human_acceptance_checklist.json`

dosyasını doldurup script’i yeniden çalıştırın (önce mevcut output root’u archive edin).
