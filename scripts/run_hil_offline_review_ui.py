#!/usr/bin/env python3
"""Safe local launch entrypoint for HIL offline review UI.

Binds only to 127.0.0.1. Does not load ReID models or open public tunnels.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
APP = SRC / "football_analytics" / "reid" / "hil_ui" / "streamlit_app.py"
STREAMLIT_CONFIG_DIR = PROJECT / "configs" / "reid" / "hil_ui" / ".streamlit"
DEFAULT_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "football-hil-ui" / "bin" / "python"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_python(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env_override = os.environ.get("HIL_UI_PYTHON")
    if env_override:
        return Path(env_override)
    if DEFAULT_ENV_PYTHON.is_file():
        return DEFAULT_ENV_PYTHON
    # Fallback: current interpreter (must already have streamlit)
    return Path(sys.executable)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch HIL offline review UI (loopback only)")
    parser.add_argument("--review-package", required=True, help="Path to review_package.json")
    parser.add_argument("--python", default=None, help="Isolated env python path")
    parser.add_argument("--port", type=int, default=0, help="Port (0 = ephemeral)")
    parser.add_argument("--address", default="127.0.0.1")
    args = parser.parse_args()

    sys.path.insert(0, str(SRC))
    from football_analytics.reid.hil_ui.security import streamlit_cli_args, validate_bind_address

    validate_bind_address(args.address)
    package = Path(args.review_package).expanduser().resolve()
    if not package.is_file():
        print(f"ERROR: review package not found: {package}", file=sys.stderr)
        return 2

    py = _resolve_python(args.python)
    if not py.is_file():
        print(
            f"ERROR: isolated UI python not found: {py}\n"
            "Create env football-hil-ui from configs/reid/hil_ui/environment.yml",
            file=sys.stderr,
        )
        return 2

    # Refuse if this is sn-reid-cpu / football-cv by name heuristic
    py_s = str(py)
    for forbidden in ("sn-reid-cpu", "football-cv"):
        if forbidden in py_s:
            print(f"ERROR: refusing to launch with protected env ({forbidden})", file=sys.stderr)
            return 2

    port = args.port or _free_port()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["HIL_REVIEW_PACKAGE"] = str(package)
    env["STREAMLIT_CONFIG_DIR"] = str(STREAMLIT_CONFIG_DIR)
    # Network not required; keep user network as-is but do not enable share.
    cmd = [str(py), "-m"] + streamlit_cli_args(app_path=str(APP), port=port, address=args.address)
    print("Launching HIL offline review UI")
    print(f"  python: {py}")
    print(f"  package: {package}")
    print(f"  bind: {args.address}:{port}")
    print("  models: not loaded")
    print("  public share/tunnel: false")
    try:
        return subprocess.call(cmd, cwd=str(PROJECT), env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
