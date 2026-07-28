#!/usr/bin/env bash
# Relaunch short-video HIL UI on host loopback (no detection/tracking rerun).
set -euo pipefail
ROOT="/home/enesturkoglu2/projects/football-analytics"
RUN="$ROOT/outputs/reid/product_new_short_video_preprocess_validation/sv_run_20260727T234854Z"
PKG="$RUN/product_review_package/review_package.json"
LOG="$RUN/ui_relaunch.log"
PY="/home/enesturkoglu2/miniconda3/envs/football-hil-ui/bin/python"
cd "$ROOT"

# Stop previous short-video UI only
pkill -f 'run_hil_offline_review_ui.py.*sv_run_20260727T234854Z' 2>/dev/null || true
sleep 1

PORT="$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
{
  echo "==== HOST UI RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "package=$PKG"
  echo "port=$PORT"
  echo "bind=127.0.0.1"
} | tee -a "$LOG"

setsid "$PY" scripts/run_hil_offline_review_ui.py \
  --review-package "$PKG" \
  --port "$PORT" \
  --address 127.0.0.1 \
  >>"$LOG" 2>&1 </dev/null &
echo $! >"$RUN/ui.pid"
LPID=$(cat "$RUN/ui.pid")

for _ in $(seq 1 40); do
  if grep -q "You can now view your Streamlit app" "$LOG"; then
    break
  fi
  sleep 0.25
done
sleep 1
STPID=$(pgrep -P "$LPID" -f 'streamlit run' | head -1 || true)

"$PY" - <<PY
import json
from pathlib import Path
run = Path("$RUN")
payload = {
  "url": f"http://127.0.0.1:$PORT",
  "port": int("$PORT"),
  "launcher_pid": int("$LPID"),
  "streamlit_pid": int("$STPID") if str("$STPID").isdigit() else None,
  "bind": "127.0.0.1",
  "match_id": "match_short_video_f2f6d8a077ca",
  "analysis_run_id": "sv_run_20260727T234854Z",
  "target_id": "target_001",
  "log": str(run / "ui_relaunch.log"),
  "status": "COMPLETED_SHORT_VIDEO_UI_RELAUNCHED_USER_ACTION_REQUIRED",
  "detection_rerun": False,
  "tracking_rerun": False,
  "logs_mutated": False,
}
(run / "ui_session.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "URL=http://127.0.0.1:$PORT"
curl -s -o /dev/null -w "http=%{http_code}\n" --max-time 5 "http://127.0.0.1:${PORT}/" || true
ss -ltnp | grep ":$PORT " || true
