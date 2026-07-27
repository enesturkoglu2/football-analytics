# MEHIL-R1 — Video review + Match Gallery render repair

Root cause: Streamlit **1.37.1** `st.image` accepts `use_column_width`, not `use_container_width`.

Fix: `football_analytics.reid.hil_ui.compat.streamlit_image`.

Gallery approvals are fail-closed unless crop is visible + SHA-verified.

```bash
PYTHONPATH=src python scripts/run_reid_mehil_r1_video_gallery_repair.py
```
