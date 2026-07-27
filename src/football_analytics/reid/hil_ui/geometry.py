"""Frame/UI coordinate transforms and click-to-bbox resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LetterboxParams:
    frame_w: int
    frame_h: int
    display_w: int
    display_h: int
    scale: float
    pad_x: float
    pad_y: float
    content_w: float
    content_h: float


@dataclass(frozen=True)
class BBoxHit:
    bbox_id: str
    bbox_xyxy: tuple[float, float, float, float]
    area: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ClickResolution:
    status: str  # hit | ambiguous | miss
    frame_x: float
    frame_y: float
    selected: BBoxHit | None
    overlapping: tuple[BBoxHit, ...]
    message: str


def letterbox_params(
    *,
    frame_w: int,
    frame_h: int,
    display_w: int,
    display_h: int,
) -> LetterboxParams:
    if frame_w <= 0 or frame_h <= 0 or display_w <= 0 or display_h <= 0:
        raise ValueError("frame and display dimensions must be positive")
    scale = min(display_w / frame_w, display_h / frame_h)
    content_w = frame_w * scale
    content_h = frame_h * scale
    pad_x = (display_w - content_w) / 2.0
    pad_y = (display_h - content_h) / 2.0
    return LetterboxParams(
        frame_w=frame_w,
        frame_h=frame_h,
        display_w=display_w,
        display_h=display_h,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        content_w=content_w,
        content_h=content_h,
    )


def ui_to_frame_coords(
    ui_x: float,
    ui_y: float,
    params: LetterboxParams,
) -> tuple[float, float] | None:
    """Map a UI click into original frame coordinates. None if outside content."""
    x = (ui_x - params.pad_x) / params.scale
    y = (ui_y - params.pad_y) / params.scale
    if x < 0 or y < 0 or x > params.frame_w or y > params.frame_h:
        return None
    return x, y


def _area(xyxy: Sequence[float]) -> float:
    return max(0.0, float(xyxy[2]) - float(xyxy[0])) * max(
        0.0, float(xyxy[3]) - float(xyxy[1])
    )


def _contains(xyxy: Sequence[float], x: float, y: float) -> bool:
    return float(xyxy[0]) <= x <= float(xyxy[2]) and float(xyxy[1]) <= y <= float(xyxy[3])


def resolve_click_to_bbox(
    *,
    ui_x: float,
    ui_y: float,
    params: LetterboxParams,
    bboxes: Sequence[Mapping[str, Any]],
) -> ClickResolution:
    """Deterministic click→bbox: smallest-area among containing boxes; never nearest-miss."""
    mapped = ui_to_frame_coords(ui_x, ui_y, params)
    if mapped is None:
        return ClickResolution(
            status="miss",
            frame_x=ui_x,
            frame_y=ui_y,
            selected=None,
            overlapping=(),
            message="click outside letterboxed frame content",
        )
    fx, fy = mapped
    hits: list[BBoxHit] = []
    for row in bboxes:
        xyxy = row["bbox_xyxy"]
        if not _contains(xyxy, fx, fy):
            continue
        hits.append(
            BBoxHit(
                bbox_id=str(row.get("bbox_id") or row.get("candidate_id") or row.get("segment_id")),
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                area=_area(xyxy),
                provenance=dict(row.get("provenance") or {}),
            )
        )
    if not hits:
        return ClickResolution(
            status="miss",
            frame_x=fx,
            frame_y=fy,
            selected=None,
            overlapping=(),
            message="no bbox contains click; nearest auto-select disabled",
        )
    hits_sorted = tuple(sorted(hits, key=lambda h: (h.area, h.bbox_id)))
    if len(hits_sorted) == 1:
        return ClickResolution(
            status="hit",
            frame_x=fx,
            frame_y=fy,
            selected=hits_sorted[0],
            overlapping=hits_sorted,
            message="single bbox hit",
        )
    # Deterministic nested/overlap policy: smallest area, then bbox_id.
    return ClickResolution(
        status="hit",
        frame_x=fx,
        frame_y=fy,
        selected=hits_sorted[0],
        overlapping=hits_sorted,
        message="overlapping bboxes; selected smallest area",
    )


def scale_only_params(*, frame_w: int, frame_h: int, display_w: int, display_h: int) -> LetterboxParams:
    """Non-letterbox uniform scale filling width or height without pad (stretch-free fit)."""
    return letterbox_params(
        frame_w=frame_w, frame_h=frame_h, display_w=display_w, display_h=display_h
    )
