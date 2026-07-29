"""Bounded review-window indexing for golden-clip UI (no full-manifest browser payload)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_WINDOW_RADIUS_FRAMES = 45  # ~1.5s @ 30fps


def load_dense_observations_index(path: Path) -> dict[str, Any]:
    """Load dense timeline once; caller should cache the result."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    obs = payload.get("observations_by_frame") or {}
    return {
        "schema_version": payload.get("schema_version"),
        "fps": float(payload.get("fps") or 30.0),
        "frame_count": int(payload.get("frame_count") or 0),
        "resolution": payload.get("resolution") or {},
        "observations_by_frame": {
            str(k): [
                {
                    "bbox_xyxy": r["bbox_xyxy"],
                    "segment_id": r.get("segment_id"),
                    "raw_track_id": str(r.get("raw_track_id")),
                    "external_candidate_code": r.get("external_candidate_code"),
                    "candidate_id": r.get("candidate_id"),
                    "detection_id": r.get("detection_id"),
                    "selectable": r.get("selectable", True),
                }
                for r in rows
                if r.get("selectable", True)
            ]
            for k, rows in obs.items()
        },
        "source_bytes": Path(path).stat().st_size,
    }


def slice_window(
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    center_frame: int,
    radius: int = DEFAULT_WINDOW_RADIUS_FRAMES,
    frame_count: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return only frames in [center-radius, center+radius]."""
    lo = max(0, int(center_frame) - int(radius))
    hi = int(center_frame) + int(radius)
    if frame_count is not None:
        hi = min(hi, int(frame_count) - 1)
    out: dict[str, list[dict[str, Any]]] = {}
    for fi in range(lo, hi + 1):
        key = str(fi)
        rows = observations_by_frame.get(key)
        if rows:
            out[key] = [dict(r) for r in rows]
    return out


def payload_byte_size(observations_by_frame: Mapping[str, Any]) -> int:
    return len(json.dumps(observations_by_frame, separators=(",", ":"), ensure_ascii=False).encode())


def track_span_lightweight(
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    raw_track_id: str,
    segment_id: str | None = None,
    sample_every: int = 15,
) -> dict[str, Any]:
    """Compute track span without copying every bbox into session_state.

    Returns sparse bbox samples for UI preview only.
    """
    first = last = None
    samples: list[dict[str, Any]] = []
    det_ids: list[str] = []
    for fi_s, rows in observations_by_frame.items():
        for r in rows:
            hit = False
            if segment_id and r.get("segment_id") == segment_id:
                hit = True
            elif str(r.get("raw_track_id")) == str(raw_track_id):
                hit = True
            if not hit:
                continue
            fi = int(fi_s)
            first = fi if first is None else min(first, fi)
            last = fi if last is None else max(last, fi)
            if r.get("detection_id"):
                det_ids.append(str(r["detection_id"]))
            if sample_every <= 1 or fi % sample_every == 0:
                samples.append(
                    {
                        "frame_index": fi,
                        "bbox_xyxy": list(r["bbox_xyxy"]),
                        "detection_id": r.get("detection_id"),
                        "raw_track_id": str(r.get("raw_track_id")),
                        "segment_id": r.get("segment_id"),
                    }
                )
    # always include ends
    if first is not None and last is not None and first != last:
        for edge in (first, last):
            rows = observations_by_frame.get(str(edge)) or []
            for r in rows:
                if str(r.get("raw_track_id")) == str(raw_track_id) or (
                    segment_id and r.get("segment_id") == segment_id
                ):
                    item = {
                        "frame_index": edge,
                        "bbox_xyxy": list(r["bbox_xyxy"]),
                        "detection_id": r.get("detection_id"),
                        "raw_track_id": str(r.get("raw_track_id")),
                        "segment_id": r.get("segment_id"),
                    }
                    if item not in samples:
                        samples.append(item)
                    break
    return {
        "first_frame": first,
        "last_frame": last,
        "sparse_bbox_observations": samples,
        "detection_ids_sample": det_ids[:50],
        "detection_id_count": len(det_ids),
    }
