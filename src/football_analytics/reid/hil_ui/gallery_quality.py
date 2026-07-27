"""Gallery crop quality audit (evidence-based labels; no invented ReID thresholds)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil_ui.gallery_view import validate_gallery_crop_for_display


QUALITY_CLASSES = (
    "USABLE_GALLERY_CANDIDATE",
    "LOW_RESOLUTION",
    "BLURRED",
    "CLIPPED",
    "OCCLUDED",
    "CONTAMINATED",
    "INVALID",
)


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def audit_gallery_crop_quality(
    crop: Mapping[str, Any],
    *,
    source_frame_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Classify one crop from decode + visual evidence (no upscaling salvage)."""
    base = validate_gallery_crop_for_display(crop)
    report: dict[str, Any] = {
        **base,
        "source_frame": crop.get("frame_index"),
        "source_segment_id": crop.get("segment_id"),
        "source_raw_track_id": crop.get("raw_track_id"),
        "crop_sha256": crop.get("crop_sha256"),
        "player_pixel_height": None,
        "blur_laplacian_variance": None,
        "clipping": False,
        "visibility": crop.get("visibility") or {},
        "occlusion": bool((crop.get("visibility") or {}).get("occluded")),
        "contamination": bool(
            (crop.get("contamination") or {}).get("multi_person")
            or (crop.get("provenance") or {}).get("contaminated")
        ),
        "quality_class": "INVALID",
        "approval_enabled": False,
        "requires_calibration": True,
        "note": "Do not approve blurry or unclear crops. A weak gallery can make candidate ranking worse.",
    }
    if not base.get("visible"):
        report["quality_class"] = "INVALID"
        return report

    path = Path(str(crop["crop_path"]))
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        report["error"] = "cv2 decode failed"
        report["quality_class"] = "INVALID"
        return report
    h, w = bgr.shape[:2]
    report["width"] = int(w)
    report["height"] = int(h)
    report["player_pixel_height"] = int(h)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    report["blur_laplacian_variance"] = _laplacian_variance(gray)

    # Clipping evidence: original bbox relative to source frame if provided
    bbox = crop.get("bbox_xyxy")
    if source_frame_size and isinstance(bbox, list) and len(bbox) == 4:
        fw, fh = source_frame_size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x1 <= 1 or y1 <= 1 or x2 >= fw - 2 or y2 >= fh - 2:
            report["clipping"] = True

    if report["contamination"]:
        report["quality_class"] = "CONTAMINATED"
        return report
    if report["occlusion"]:
        report["quality_class"] = "OCCLUDED"
        return report
    if report["clipping"]:
        report["quality_class"] = "CLIPPED"
        return report
    # Size class from the crop itself (factual), not a ReID score threshold.
    if min(w, h) < 48 or h < 48:
        report["quality_class"] = "LOW_RESOLUTION"
        return report

    # Blur label deferred to set-relative pass
    report["quality_class"] = "PENDING_BLUR_RELATIVE"
    return report


def finalize_blur_relative(
    audits: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mark clearly softer crops BLURRED relative to the set median (not a fixed product cut)."""
    pending = [
        a for a in audits if a.get("quality_class") == "PENDING_BLUR_RELATIVE"
    ]
    vars_ = [
        float(a["blur_laplacian_variance"])
        for a in pending
        if a.get("blur_laplacian_variance") is not None
    ]
    median = float(np.median(vars_)) if vars_ else 0.0
    out: list[dict[str, Any]] = []
    for a in audits:
        row = dict(a)
        if row.get("quality_class") == "PENDING_BLUR_RELATIVE":
            blur = float(row.get("blur_laplacian_variance") or 0.0)
            # Relative evidence: much softer than cohort median → BLURRED
            if median > 0 and blur < 0.35 * median:
                row["quality_class"] = "BLURRED"
                row["blur_relative_to_median"] = blur / median
            else:
                row["quality_class"] = "USABLE_GALLERY_CANDIDATE"
                row["blur_relative_to_median"] = (blur / median) if median else None
                row["approval_enabled"] = True
        # Disable approval for non-usable classes
        if row["quality_class"] != "USABLE_GALLERY_CANDIDATE":
            row["approval_enabled"] = False
        out.append(row)
    return out


def audit_gallery_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_frame_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    raw = [
        audit_gallery_crop_quality(c, source_frame_size=source_frame_size)
        for c in candidates
    ]
    finalized = finalize_blur_relative(raw)
    usable = [a for a in finalized if a["quality_class"] == "USABLE_GALLERY_CANDIDATE"]
    return {
        "schema_version": "mehil_r2_gallery_quality_audit_v1",
        "candidate_count": len(finalized),
        "usable_count": len(usable),
        "audits": finalized,
        "fixed_reid_threshold_invented": False,
        "upscaling_used": False,
        "user_message": (
            "Do not approve blurry or unclear crops. A weak gallery can make candidate ranking worse."
        ),
    }
