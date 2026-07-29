#!/usr/bin/env python3
"""Launch Target Ground Truth annotation UI (loopback only; no product log writes)."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
APP = SRC / "football_analytics" / "reid" / "golden_clip" / "streamlit_app.py"
STREAMLIT_CONFIG_DIR = PROJECT / "configs" / "reid" / "hil_ui" / ".streamlit"
DEFAULT_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "football-hil-ui" / "bin" / "python"
DEFAULT_ROOT = (
    PROJECT
    / "outputs/reid/target_golden_clip_r1/match_short_video_f2f6d8a077ca/sv_run_20260727T234854Z"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--address", default="127.0.0.1")
    args = parser.parse_args()

    sys.path.insert(0, str(SRC))
    from football_analytics.reid.hil_ui.security import streamlit_cli_args, validate_bind_address

    validate_bind_address(args.address)
    root = args.golden_root.expanduser().resolve()
    if not (root / "read_only_refs" / "artifact_pointers.json").is_file():
        print(f"ERROR: golden root not ready: {root}", file=sys.stderr)
        return 2

    py = Path(args.python) if args.python else DEFAULT_ENV_PYTHON
    if not py.is_file():
        py = Path(sys.executable)
    port = args.port or _free_port()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["GOLDEN_CLIP_ROOT"] = str(root)
    env["STREAMLIT_CONFIG_DIR"] = str(STREAMLIT_CONFIG_DIR)
    # Ensure product package env is NOT set (no contamination)
    env.pop("HIL_REVIEW_PACKAGE", None)

    cmd = [str(py), "-m"] + streamlit_cli_args(app_path=str(APP), port=port, address=args.address)
    print("Launching Target Ground Truth UI")
    print(f"  root: {root}")
    print(f"  bind: {args.address}:{port}")
    print("  product_decision_log: not used")
    try:
        return subprocess.call(cmd, cwd=str(PROJECT), env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
