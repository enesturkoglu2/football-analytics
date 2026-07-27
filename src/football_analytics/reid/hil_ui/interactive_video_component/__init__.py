"""Local Streamlit custom component: interactive video + canvas bbox overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_COMPONENT_DIR = Path(__file__).parent / "frontend"


def interactive_video_review(
    *,
    video_data_url: str,
    observations_by_frame: dict,
    markers: list | None = None,
    selected_raw_track_id: str | None = None,
    selected_segment_id: str | None = None,
    track_end_frame: int | None = None,
    fps: float = 30.0,
    video_width: int = 960,
    video_height: int = 540,
    key: str | None = None,
    default: Any = None,
):
    """Local offline interactive HTML5 video + canvas bbox overlay.

    Returns dict events from the browser, e.g.
    ``{"type":"bbox_click","frame_index":12,"raw_track_id":"11",...}``.
    """
    import streamlit.components.v1 as components

    component = components.declare_component(
        "hil_interactive_video",
        path=str(_COMPONENT_DIR),
    )
    return component(
        video_data_url=video_data_url,
        observations_by_frame=observations_by_frame or {},
        markers=markers or [],
        selected_raw_track_id=selected_raw_track_id,
        selected_segment_id=selected_segment_id,
        track_end_frame=track_end_frame,
        fps=float(fps),
        video_width=int(video_width),
        video_height=int(video_height),
        key=key,
        default=default,
    )
