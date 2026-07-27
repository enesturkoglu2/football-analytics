"""Local Streamlit server security helpers."""

from __future__ import annotations

from typing import Any

ALLOWED_BIND_ADDRESS = "127.0.0.1"
FORBIDDEN_BIND_ADDRESSES = {"0.0.0.0", "::", "[::]"}


def validate_bind_address(address: str) -> str:
    text = str(address).strip()
    if text in FORBIDDEN_BIND_ADDRESSES:
        raise ValueError(f"public bind address forbidden: {text}")
    if text != ALLOWED_BIND_ADDRESS:
        raise ValueError(f"bind address must be {ALLOWED_BIND_ADDRESS}, got {text!r}")
    return text


def streamlit_cli_args(
    *,
    app_path: str,
    port: int,
    address: str = ALLOWED_BIND_ADDRESS,
) -> list[str]:
    validate_bind_address(address)
    if not (1 <= int(port) <= 65535):
        raise ValueError("port out of range")
    return [
        "streamlit",
        "run",
        app_path,
        "--server.address",
        address,
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "true",
        "--server.fileWatcherType",
        "none",
    ]


def security_audit_payload(
    *,
    address: str,
    port: int,
    gather_usage_stats: bool = False,
    headless: bool = True,
    public_share: bool = False,
    public_tunnel: bool = False,
) -> dict[str, Any]:
    validate_bind_address(address)
    if gather_usage_stats:
        raise ValueError("usage stats must be disabled")
    if public_share or public_tunnel:
        raise ValueError("public share/tunnel forbidden")
    return {
        "schema_version": "hil_b_streamlit_server_security_audit_v1",
        "bind_address": address,
        "port": port,
        "headless": headless,
        "gatherUsageStats": False,
        "public_share": False,
        "public_tunnel": False,
        "telemetry": False,
        "loopback_only": True,
    }
