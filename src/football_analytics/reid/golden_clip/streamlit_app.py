"""Streamlit Target Ground Truth UI — lightweight failure-window pilot (R1.1).

Does NOT send full dense manifest or re-encode video on every click/rerun.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

# Heavy model imports intentionally absent.


def _ensure_src_path() -> None:
    import sys

    src = Path(__file__).resolve().parents[3]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_path()

from football_analytics.reid.golden_clip import UI_PROFILE  # noqa: E402
from football_analytics.reid.golden_clip.annotation_log import (  # noqa: E402
    AnnotationLog,
    assert_not_product_log,
)
from football_analytics.reid.golden_clip.pilot import (  # noqa: E402
    COVERAGE_SCOPE,
    NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
    PILOT_LABELS,
    build_pilot_label_event,
    choose_pilot_next_gate,
    empty_pilot_doc,
    generate_failure_windows,
    summarize_pilot_labels,
)
from football_analytics.reid.golden_clip.schema import new_session_id  # noqa: E402
from football_analytics.reid.golden_clip.window_index import (  # noqa: E402
    DEFAULT_WINDOW_RADIUS_FRAMES,
    load_dense_observations_index,
    payload_byte_size,
    slice_window,
    track_span_lightweight,
)
from football_analytics.reid.hil_ui.interactive_video_component import (  # noqa: E402
    interactive_video_review,
)
from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: E402
    ensure_interactive_review_proxy,
)


def _root_from_env() -> Path:
    raw = os.environ.get("GOLDEN_CLIP_ROOT")
    if not raw:
        raise RuntimeError("GOLDEN_CLIP_ROOT env var required")
    return Path(raw).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Target Ground Truth Pilot", layout="wide")
    st.title("Target Ground Truth — Failure Window Pilot")
    st.caption(
        f"ui_profile={UI_PROFILE} · coverage_scope={COVERAGE_SCOPE} · "
        "product decision/approval loglarına yazılmaz"
    )
    st.warning(
        "R1.1: Tam dense manifest tarayıcıya gönderilmez. "
        "Click donması giderildi; önce failure-window pilot annotation."
    )

    root = _root_from_env()
    refs = _load_json(root / "read_only_refs" / "artifact_pointers.json")
    session_dir = root / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ann_path = session_dir / "annotation_log.jsonl"
    assert_not_product_log(ann_path)
    meta_path = session_dir / "annotation_session.json"
    pilot_path = session_dir / "failure_window_pilot.json"
    instr_path = session_dir / "ui_instrumentation.jsonl"

    @st.cache_data(show_spinner=False)
    def _cached_index(dense_path: str, mtime: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        idx = load_dense_observations_index(Path(dense_path))
        idx["deserialize_sec"] = time.perf_counter() - t0
        return idx

    @st.cache_data(show_spinner=False)
    def _cached_proxy(video: str, sha: str, out: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        info = ensure_interactive_review_proxy(
            source_video=Path(video),
            source_video_sha256=sha,
            output_path=Path(out),
        )
        info["encode_or_read_sec"] = time.perf_counter() - t0
        # Keep only small metadata in return for session; data_url cached by Streamlit
        return info

    if "gt_session" not in st.session_state:
        if meta_path.is_file():
            st.session_state.gt_session = _load_json(meta_path)
        else:
            sess = {
                "annotation_session_id": new_session_id(),
                "reviewer": "golden_clip_reviewer",
                "source_video_sha256": refs["source_video_sha256"],
                "match_id": refs["match_id"],
                "analysis_run_id": refs["analysis_run_id"],
                "target_id": refs["target_id"],
                "fps": refs["fps"],
                "frame_count": refs["frame_count"],
                "mode": "TARGET_FAILURE_WINDOW_PILOT",
            }
            _write_json(meta_path, sess)
            st.session_state.gt_session = sess
    for k, default in (
        ("selection", {}),
        ("proposal", None),
        ("failure_windows", []),
        ("window_idx", 0),
        ("processed_event_uuids", []),  # list for Streamlit session serialize
        ("video_url_sent", False),
        ("center_frame", 0),
        ("pending_seek_frame", 0),  # initial seek; cleared after apply
        ("pending_force_pause", False),
        ("ui_mode", "interactive"),  # or fallback
        ("rerun_count", 0),
        ("last_click_latency_ms", None),
        ("click_t0", None),
    ):
        if k not in st.session_state:
            st.session_state[k] = default

    st.session_state.rerun_count = int(st.session_state.rerun_count) + 1
    sess = st.session_state.gt_session
    fps = float(sess["fps"])
    frame_count = int(sess["frame_count"])
    log = AnnotationLog(ann_path)

    dense_path = Path(refs["dense_bbox_timeline"])
    if not dense_path.is_file():
        dense_path = _project_root() / refs["dense_bbox_timeline"]
    idx = _cached_index(str(dense_path), dense_path.stat().st_mtime)
    obs_all = idx["observations_by_frame"]

    video = Path(refs["source_video"])
    if not video.is_file():
        video = _project_root() / refs["source_video"]
    proxy_path = session_dir / "interactive_review" / "source_proxy_960.mp4"
    proxy = _cached_proxy(str(video), refs["source_video_sha256"], str(proxy_path))

    # Video data URL only on first component mount in this browser session
    video_data_url = proxy["data_url"] if not st.session_state.video_url_sent else ""
    if not st.session_state.video_url_sent:
        st.session_state.video_url_sent = True

    center = int(st.session_state.center_frame)
    window_obs = slice_window(
        obs_all,
        center_frame=center,
        radius=DEFAULT_WINDOW_RADIUS_FRAMES,
        frame_count=frame_count,
    )
    win_bytes = payload_byte_size(window_obs)

    # Instrumentation (append-only, tiny)
    with instr_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "rerun_count": st.session_state.rerun_count,
                    "center_frame": center,
                    "window_payload_bytes": win_bytes,
                    "full_manifest_bytes_not_sent": True,
                    "video_data_url_resent": bool(video_data_url),
                    "proxy_bytes": proxy.get("bytes"),
                    "dense_deserialize_sec_cached": idx.get("deserialize_sec"),
                    "ui_mode": st.session_state.ui_mode,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", st.session_state.ui_mode)
    c2.metric("Window payload", f"{win_bytes/1024:.1f} KB")
    c3.metric("Rerun count", st.session_state.rerun_count)
    c4.metric(
        "Click latency ms",
        st.session_state.last_click_latency_ms
        if st.session_state.last_click_latency_ms is not None
        else "—",
    )

    mode = st.radio(
        "UI mode",
        ["interactive", "fallback"],
        index=0 if st.session_state.ui_mode == "interactive" else 1,
        horizontal=True,
    )
    st.session_state.ui_mode = mode

    sel = st.session_state.selection or {}

    def _handle_click(click: dict[str, Any]) -> None:
        eid = str(click.get("event_uuid") or "")
        processed = list(st.session_state.processed_event_uuids or [])
        if eid and eid in processed:
            return
        if eid:
            processed.append(eid)
            # Bound growth
            st.session_state.processed_event_uuids = processed[-200:]
        t0 = st.session_state.click_t0
        if t0:
            st.session_state.last_click_latency_ms = int((time.perf_counter() - t0) * 1000)
        st.session_state.selection = {
            "selected_raw_track_id": str(click.get("raw_track_id")),
            "selected_segment_id": click.get("segment_id"),
            "selected_frame_index": int(click.get("frame_index")),
            "selected_bbox_xyxy": click.get("bbox_xyxy"),
            "detection_id": click.get("detection_id"),
            "event_uuid": eid,
        }
        st.session_state.center_frame = int(click.get("frame_index") or 0)
        st.session_state.pending_seek_frame = st.session_state.center_frame
        st.session_state.pending_force_pause = True
        # Lightweight proposal — no full track bbox dump into session_state
        span = track_span_lightweight(
            obs_all,
            raw_track_id=str(click.get("raw_track_id")),
            segment_id=click.get("segment_id"),
            sample_every=30,
        )
        st.session_state.proposal = {
            "raw_track_id": str(click.get("raw_track_id")),
            "segment_id": click.get("segment_id"),
            "first_frame": span["first_frame"],
            "last_frame": span["last_frame"],
            "sparse_bbox_observations": span["sparse_bbox_observations"][:20],
            "detection_id_count": span["detection_id_count"],
        }
        if span["first_frame"] is not None and span["last_frame"] is not None:
            st.session_state.failure_windows = generate_failure_windows(
                observations_by_frame=obs_all,
                selected_raw_track_id=str(click.get("raw_track_id")),
                selected_segment_id=click.get("segment_id"),
                track_first=int(span["first_frame"]),
                track_last=int(span["last_frame"]),
                fps=fps,
                frame_count=frame_count,
            )
            st.session_state.window_idx = 0
            if st.session_state.failure_windows:
                st.session_state.center_frame = int(
                    st.session_state.failure_windows[0]["center_frame"]
                )
                st.session_state.pending_seek_frame = st.session_state.center_frame

    if mode == "interactive":
        st.markdown("### Interactive video (bounded window)")
        st.caption(
            f"Gönderilen gözlem penceresi: ±{DEFAULT_WINDOW_RADIUS_FRAMES} frame "
            f"(@ center={center}). Full dense manifest gönderilmez."
        )
        track_end = None
        if st.session_state.proposal and st.session_state.proposal.get("last_frame") is not None:
            track_end = int(st.session_state.proposal["last_frame"])
        markers = [
            {
                "event_id": w["window_id"],
                "label": f"{w['kind'][:16]}@{w['center_frame']}",
                "frame_index": int(w["center_frame"]),
            }
            for w in (st.session_state.failure_windows or [])[:12]
        ]
        st.session_state.click_t0 = time.perf_counter()
        seek_once = st.session_state.pending_seek_frame
        if seek_once is not None:
            st.session_state.pending_seek_frame = None
        force_once = bool(st.session_state.pending_force_pause)
        if force_once:
            st.session_state.pending_force_pause = False
        click = interactive_video_review(
            video_data_url=video_data_url,
            observations_by_frame=window_obs,
            markers=markers,
            selected_raw_track_id=sel.get("selected_raw_track_id"),
            selected_segment_id=sel.get("selected_segment_id"),
            track_end_frame=track_end,
            fps=fps,
            video_width=int(refs["resolution"]["width"]),
            video_height=int(refs["resolution"]["height"]),
            force_pause=force_once,
            seek_frame=seek_once,
            key="gt_pilot_interactive_stable",
        )
        if isinstance(click, dict) and click.get("type") == "bbox_click":
            _handle_click(click)
            st.success(
                f"Seçildi · track={click.get('raw_track_id')} · "
                f"frame={click.get('frame_index')} · uuid={click.get('event_uuid')}"
            )
            # Soft refresh without remounting video: update widgets only
            st.rerun()
        elif isinstance(click, dict) and click.get("type") == "marker_seek":
            st.session_state.center_frame = int(click.get("frame_index") or 0)
            st.rerun()
    else:
        st.markdown("### Fallback — static frame + candidate list")
        st.caption("Dense timeline browser’a yüklenmez; tek frame decode.")
        fi = st.number_input(
            "Frame",
            min_value=0,
            max_value=max(0, frame_count - 1),
            value=int(center),
        )
        st.session_state.center_frame = int(fi)
        # Decode one frame
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        cap.release()
        rows = list(obs_all.get(str(int(fi))) or [])
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for r in rows:
                x1, y1, x2, y2 = [int(round(v)) for v in r["bbox_xyxy"]]
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 220, 220), 2)
                cv2.putText(
                    rgb,
                    f"{r.get('raw_track_id')}",
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 220),
                    1,
                )
            st.image(Image.fromarray(rgb), use_container_width=True)
        options = [
            f"{r.get('raw_track_id')} | {r.get('segment_id')} | {r.get('bbox_xyxy')}"
            for r in rows
        ]
        pick = st.selectbox("BBox candidates on frame", options=["(none)"] + options)
        if pick != "(none)" and st.button("Select candidate (fallback)", type="primary"):
            r = rows[options.index(pick)]
            _handle_click(
                {
                    "type": "bbox_click",
                    "event_uuid": f"fb_{uuid.uuid4().hex[:12]}",
                    "frame_index": int(fi),
                    "raw_track_id": str(r.get("raw_track_id")),
                    "segment_id": r.get("segment_id"),
                    "detection_id": r.get("detection_id"),
                    "bbox_xyxy": r.get("bbox_xyxy"),
                }
            )
            st.rerun()

    st.markdown("### Current selection / proposal")
    st.write(
        {
            "selection": st.session_state.selection,
            "proposal_span": (
                {
                    "raw_track_id": (st.session_state.proposal or {}).get("raw_track_id"),
                    "first_frame": (st.session_state.proposal or {}).get("first_frame"),
                    "last_frame": (st.session_state.proposal or {}).get("last_frame"),
                    "detection_id_count": (st.session_state.proposal or {}).get(
                        "detection_id_count"
                    ),
                }
                if st.session_state.proposal
                else None
            ),
        }
    )

    windows = st.session_state.failure_windows or []
    if windows:
        st.markdown("### Failure windows")
        idx_w = int(st.session_state.window_idx)
        idx_w = max(0, min(idx_w, len(windows) - 1))
        st.session_state.window_idx = idx_w
        w = windows[idx_w]
        nav1, nav2, nav3 = st.columns(3)
        if nav1.button("Previous window") and idx_w > 0:
            st.session_state.window_idx = idx_w - 1
            st.session_state.center_frame = int(windows[idx_w - 1]["center_frame"])
            st.session_state.pending_seek_frame = st.session_state.center_frame
            st.rerun()
        nav2.write(f"Window {idx_w + 1}/{len(windows)} · {w['kind']}")
        if nav3.button("Next window") and idx_w < len(windows) - 1:
            st.session_state.window_idx = idx_w + 1
            st.session_state.center_frame = int(windows[idx_w + 1]["center_frame"])
            st.session_state.pending_seek_frame = st.session_state.center_frame
            st.rerun()

        st.write(
            {
                "window_id": w["window_id"],
                "center_frame": w["center_frame"],
                "time_range": [w["start_time"], w["end_time"]],
                "previous_raw_track": w["previous_raw_track_id"],
                "possible_next_raw_tracks": w["possible_next_raw_tracks"][:8],
                "time_gap_frames": w.get("time_gap_frames"),
                "detection_continuity_frames": w.get("detection_continuity_frames"),
                "bbox_overlap": w.get("bbox_overlap"),
                "displacement_px": w.get("image_position_displacement_px"),
                "tracker_variant": w.get("tracker_variant"),
            }
        )
        next_tid = st.selectbox(
            "Optional: click/select next raw track for this window",
            ["(none)"] + list(w.get("possible_next_raw_tracks") or [])[:12],
        )
        st.markdown("### Pilot labels")
        cols = st.columns(4)
        labels = list(PILOT_LABELS)
        for i, lab in enumerate(labels):
            if cols[i % 4].button(lab, key=f"lab_{w['window_id']}_{lab}"):
                ev = build_pilot_label_event(
                    window=w,
                    label=lab,
                    reviewer=sess["reviewer"],
                    annotation_session_id=sess["annotation_session_id"],
                    source_video_sha256=sess["source_video_sha256"],
                    match_id=sess["match_id"],
                    analysis_run_id=sess["analysis_run_id"],
                    target_id=sess["target_id"],
                    selected_next_raw_track_id=(
                        None if next_tid == "(none)" else str(next_tid)
                    ),
                )
                log.append(ev)
                # Update pilot summary doc
                pilot = (
                    _load_json(pilot_path)
                    if pilot_path.is_file()
                    else empty_pilot_doc(
                        source_video_sha256=sess["source_video_sha256"],
                        match_id=sess["match_id"],
                        analysis_run_id=sess["analysis_run_id"],
                        target_id=sess["target_id"],
                        annotation_session_id=sess["annotation_session_id"],
                        reviewer=sess["reviewer"],
                    )
                )
                pilot.setdefault("windows", []).append(
                    {
                        "window_id": w["window_id"],
                        "label": lab,
                        "event_id": ev["event_id"],
                        "center_frame": w["center_frame"],
                    }
                )
                events = [
                    e
                    for e in log.read_raw()
                    if e.get("action") == "PILOT_FAILURE_WINDOW_LABEL"
                ]
                summary = summarize_pilot_labels(events)
                gate = choose_pilot_next_gate(summary)
                pilot["summary"] = summary
                pilot["exact_next_gate"] = gate
                _write_json(pilot_path, pilot)
                _write_json(session_dir / "pilot_summary.json", summary)
                _write_json(session_dir / "pilot_next_gate.json", gate)
                st.success(f"Pilot label yazıldı: {lab} ({summary['labeled_window_count']} total)")
                st.rerun()
    else:
        st.info("Önce hedef oyuncunun bbox’ına tıklayın; failure window’lar üretilecek.")

    with st.expander("Advanced / honesty"):
        events = log.read_raw()
        summary = summarize_pilot_labels(events)
        st.write(summary)
        st.write(
            {
                "coverage_scope": COVERAGE_SCOPE,
                "accepted_full_gt": False,
                "full_metrics": NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
                "product_logs_path_refused": True,
                "annotation_log": str(ann_path),
            }
        )


if __name__ == "__main__":
    run_app()
