"""Gallery crop decode/SHA validation for fail-closed Match Gallery approvals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from football_analytics.reid.hil.common import sha256_file


def validate_gallery_crop_for_display(crop: Mapping[str, Any]) -> dict[str, Any]:
    """Return visibility + approval eligibility for one crop candidate."""
    path = Path(str(crop.get("crop_path") or ""))
    expected = str(crop.get("crop_sha256") or "").lower()
    result: dict[str, Any] = {
        "crop_id": crop.get("crop_id"),
        "path": str(path),
        "expected_sha256": expected,
        "exists": path.is_file(),
        "visible": False,
        "approval_enabled": False,
        "error": None,
        "width": None,
        "height": None,
        "actual_sha256": None,
    }
    if not path.is_file():
        result["error"] = f"crop file missing: {path}"
        return result
    actual = sha256_file(path)
    result["actual_sha256"] = actual
    if expected and actual != expected.lower():
        result["error"] = f"crop SHA mismatch expected={expected} actual={actual}"
        return result
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.load()
            result["width"], result["height"] = img.size
            if result["width"] <= 0 or result["height"] <= 0:
                result["error"] = "invalid image dimensions"
                return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"crop decode failed: {exc}"
        return result
    result["visible"] = True
    result["approval_enabled"] = True
    return result
