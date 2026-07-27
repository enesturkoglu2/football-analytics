"""Shared validation helpers for HIL schemas."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HilValidationError(ValueError):
    """Raised when an HIL schema or provenance contract fails."""


def require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HilValidationError(f"{field} must be a mapping")
    return value


def require_str(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise HilValidationError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise HilValidationError(f"{field} must be a non-empty string")
    return value


def require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise HilValidationError(f"{field} must be a bool")
    return value


def require_int(value: Any, *, field: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HilValidationError(f"{field} must be an int")
    if min_value is not None and value < min_value:
        raise HilValidationError(f"{field} must be >= {min_value}")
    return value


def require_sha256(value: Any, *, field: str) -> str:
    text = require_str(value, field=field).lower()
    if not _SHA256_RE.match(text):
        raise HilValidationError(f"{field} must be a 64-char lowercase hex SHA-256")
    return text


def require_sha256_or_none(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return require_sha256(value, field=field)


def validate_no_path_traversal(path_value: Any, *, field: str) -> str:
    text = require_str(path_value, field=field)
    if "\x00" in text:
        raise HilValidationError(f"{field} contains NUL")
    # Reject obvious traversal / escape patterns in stored relative or absolute refs.
    parts = Path(text).parts
    if any(part == ".." for part in parts):
        raise HilValidationError(f"{field} must not contain path traversal '..'")
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    import json

    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def reject_mutable_runtime_leak(obj: Mapping[str, Any], *, field: str) -> None:
    forbidden = {"model", "torch_module", "numpy_array", "_cache", "__dict__"}
    leaked = sorted(set(obj) & forbidden)
    if leaked:
        raise HilValidationError(
            f"{field} contains forbidden mutable runtime keys: {leaked}"
        )


SPORTSREID_MODEL_ID = "osnet_x1_0_sportsreid_soccernet"
SPORTSREID_CHECKPOINT_SHA256 = (
    "c61e0da2007f7c7f4d889cb68774dfeecf8c4c433e0bfe3858b48b8655f83e91"
)
