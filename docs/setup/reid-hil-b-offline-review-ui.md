# ReID-HIL-B — Offline Target Recovery Review UI

## Durum
İzole Streamlit UI ortamı (`football-hil-ui`) ve HIL-A decision-log entegrasyonlu offline review arayüzü.

## Güvenlik
- Bind: yalnız `127.0.0.1`
- `gatherUsageStats=false`
- Public share / tunnel yok
- ReID model / CLIP yüklenmez
- Karar logu append-only; undo = superseding kayıt

## Launch (Cursor Agent veya script)

```bash
# PYTHONPATH=src ile izole env
/home/enesturkoglu2/miniconda3/envs/football-hil-ui/bin/python \
  scripts/run_hil_offline_review_ui.py \
  --review-package path/to/review_package.json
```

veya:

```bash
HIL_UI_PYTHON=/home/enesturkoglu2/miniconda3/envs/football-hil-ui/bin/python \
  python scripts/run_hil_offline_review_ui.py \
  --review-package outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/fixture/review_package.json
```

## Environment
Spec: `configs/reid/hil_ui/environment.yml`  
Requirements: `configs/reid/hil_ui/requirements.txt`  
Streamlit config: `configs/reid/hil_ui/.streamlit/config.toml`

Ana ortamlar (`football-cv`, `sn-reid-cpu`) değiştirilmez.

## Modüller
- `src/football_analytics/reid/hil_ui/` — review package, geometry, queue, decisions service, Streamlit app
- HIL-A: `src/football_analytics/reid/hil/`

## HIL-B-R1 usability repair
- Direct click: `streamlit-image-coordinates==0.1.9` (MIT, local-only) + `Select this player`
- Selected tracklet highlight / not-visible message / sparse observation notice
- Confirmation: human summary first; raw JSON under Advanced
- Next acceptance gate: `REID_HIL_B_R2_HUMAN_ACCEPTANCE_SESSION`
