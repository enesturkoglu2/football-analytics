#!/usr/bin/env bash
# Launch Target Ground Truth UI on host loopback (detached).
set -euo pipefail
ROOT="/home/enesturkoglu2/projects/football-analytics"
GROOT="$ROOT/outputs/reid/target_golden_clip_r1/match_short_video_f2f6d8a077ca/sv_run_20260727T234854Z"
LOG="$GROOT/ui_launch.log"
PY="/home/enesturkoglu2/miniconda3/envs/football-hil-ui/bin/python"
cd "$ROOT"

pkill -f 'run_target_golden_clip_r1_ui.py' 2>/dev/null || true
sleep 1

PORT="$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
{
  echo "==== GT UI LAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "root=$GROOT"
  echo "port=$PORT"
} | tee -a "$LOG"

setsid "$PY" scripts/run_target_golden_clip_r1_ui.py \
  --golden-root "$GROOT" \
  --port "$PORT" \
  --address 127.0.0.1 \
  >>"$LOG" 2>&1 </dev/null &
echo $! >"$GROOT/ui.pid"
LPID=$(cat "$GROOT/ui.pid")

for _ in $(seq 1 50); do
  if grep -q "You can now view your Streamlit app" "$LOG"; then
    break
  fi
  sleep 0.25
done
sleep 1

"$PY" - <<PY
import json
from pathlib import Path
groot = Path("$GROOT")
payload = {
  "url": f"http://127.0.0.1:$PORT",
  "port": int("$PORT"),
  "launcher_pid": int("$LPID"),
  "bind": "127.0.0.1",
  "match_id": "match_short_video_f2f6d8a077ca",
  "analysis_run_id": "sv_run_20260727T234854Z",
  "target_id": "target_001",
  "status": "COMPLETED_TARGET_GOLDEN_CLIP_UI_READY_USER_ACTION_REQUIRED",
  "log": str(groot / "ui_launch.log"),
  "detection_rerun": False,
  "tracking_rerun": False,
  "product_logs_mutated": False,
}
(groot / "ui_session.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "URL=http://127.0.0.1:$PORT"
curl -s -o /dev/null -w "http=%{http_code}\n" --max-time 5 "http://127.0.0.1:${PORT}/" || true
