"""Streamlit entrypoint for offline HIL target recovery review (local loopback only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Heavy model imports are intentionally absent.

HELPER_WARNING = "SportsReID is a helper ranker; it does not confirm identity."


def _ensure_src_path() -> None:
    root = Path(__file__).resolve().parents[3]  # .../src
    # football_analytics lives under src/
    src_root = root if root.name == "src" else Path(__file__).resolve().parents[4] / "src"
    # __file__ = .../src/football_analytics/reid/hil_ui/streamlit_app.py → parents[3]=src
    import sys

    src = Path(__file__).resolve().parents[3]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_path()

from football_analytics.reid.hil.decisions import DecisionAction  # noqa: E402
from football_analytics.reid.hil_ui.actions import map_ui_action  # noqa: E402
from football_analytics.reid.hil_ui.candidates_view import (  # noqa: E402
    find_candidate_covering_bbox,
    sort_candidates_for_display,
    sportsreid_display_fields,
)
from football_analytics.reid.hil_ui.decisions_service import (  # noqa: E402
    correct_or_revoke,
    history_for_event,
    submit_decision,
    undo_last_decision,
)
from football_analytics.reid.hil_ui.geometry import (  # noqa: E402
    letterbox_params,
    resolve_click_to_bbox,
)
from football_analytics.reid.hil_ui.media import decode_frame_to_cache  # noqa: E402
from football_analytics.reid.hil_ui.session import open_review_session  # noqa: E402


def _package_path_from_env() -> Path:
    raw = os.environ.get("HIL_REVIEW_PACKAGE")
    if not raw:
        raise RuntimeError(
            "HIL_REVIEW_PACKAGE env var is required (set by launch script)"
        )
    return Path(raw).expanduser().resolve()


def _frame_bboxes_for_event(session, frame_index: int) -> list[dict[str, Any]]:
    man = session.current_manifest()
    rows: list[dict[str, Any]] = []
    if not man:
        return rows
    for cand in man.get("candidates", []):
        for ref in cand.get("bbox_references") or []:
            if int(ref["frame_index"]) != int(frame_index):
                # Also expose middle-frame refs when seeking nearby? Only exact frame.
                continue
            rows.append(
                {
                    "bbox_id": f"{cand['candidate_id']}:{ref['frame_index']}",
                    "candidate_id": cand["candidate_id"],
                    "segment_id": cand["segment_id"],
                    "raw_track_id": cand["raw_track_id"],
                    "bbox_xyxy": list(ref["bbox_xyxy"]),
                    "provenance": {
                        "listed_candidate": True,
                        "appearance_rank": cand.get("appearance_rank"),
                        "T_max": cand.get("T_max"),
                        "D_max": cand.get("D_max"),
                        "S": cand.get("S"),
                        "sportsreid_model_id": cand.get("sportsreid_model_id"),
                        "sportsreid_checkpoint_sha256": cand.get(
                            "sportsreid_checkpoint_sha256"
                        ),
                    },
                }
            )
    # Optional extra eligible boxes from package metadata (direct-only universe)
    extras = (session.package.get("provenance") or {}).get("extra_frame_bboxes") or {}
    for item in extras.get(str(frame_index), []):
        rows.append(dict(item))
    return rows


def run_app() -> None:
    import streamlit as st
    import plotly.graph_objects as go
    from PIL import Image
    import numpy as np

    st.set_page_config(page_title="HIL Offline Review", layout="wide")
    st.title("Target Recovery Offline Review")
    st.caption("Local loopback only · no inference · append-only decisions")
    st.warning(HELPER_WARNING)

    package_path = _package_path_from_env()
    if "session" not in st.session_state:
        st.session_state.session = open_review_session(package_path)
    session = st.session_state.session

    tab_pkg, tab_queue, tab_review, tab_hist = st.tabs(
        ["Session / Package", "Recovery Queue", "Recovery Review", "Decision History"]
    )

    with tab_pkg:
        summary = session.package_summary()
        st.json(summary)
        st.write(
            "Users cannot browse arbitrary filesystem paths; package is fixed by launch env."
        )

    with tab_queue:
        filt = st.selectbox(
            "Filter",
            [
                "all",
                "unresolved",
                "deferred",
                "confirmed",
                "invalid",
                "revoked-needs-review",
            ],
        )
        rows = session.queue(filt)
        st.write(f"{len(rows)} events")
        for row in rows:
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(f"{row['event_id']} · {row['event_type']}")
            cols[1].write(row["review_status"])
            cols[2].write(f"candidates={row['candidate_count']}")
            if cols[3].button("Open", key=f"open_{row['event_id']}"):
                for i, ev in enumerate(session.events):
                    if ev["event_id"] == row["event_id"]:
                        session.set_event_index(i)
                        break

    with tab_review:
        event = session.current_event
        st.subheader(f"{event['event_id']} · {event['event_type']}")
        nav1, nav2 = st.columns(2)
        if nav1.button("Previous Event"):
            session.previous_event()
            st.rerun()
        if nav2.button("Next Event"):
            session.next_event()
            st.rerun()

        start = int(event["review_window_start_frame"])
        end = int(event["review_window_end_frame"])
        frame_index = st.slider("Frame", min_value=start, max_value=max(start, end), value=start)
        st.write(f"Current frame index: {frame_index}")

        man = session.current_manifest()
        candidates = sort_candidates_for_display(man["candidates"]) if man else []
        st.info(f"Showing all {len(candidates)} candidates (no top-k hiding).")

        bboxes = _frame_bboxes_for_event(session, frame_index)
        display_w, display_h = 640, 360
        params = None
        img_arr = None

        if session.package.get("source_video_available"):
            cache_root = (
                Path(session.package["writable_session_root_resolved"]) / "media_cache"
            )
            ro = [Path(p) for p in session.package["read_only_source_roots_resolved"]]
            decoded = decode_frame_to_cache(
                video_path=Path(session.package["source_video_resolved"]),
                video_sha256=session.package["source_video_sha256"],
                frame_index=frame_index,
                cache_root=cache_root,
                read_only_roots=ro,
                overlay_bboxes=[b["bbox_xyxy"] for b in bboxes],
            )
            img = Image.open(decoded["path"]).convert("RGB")
            frame_w, frame_h = img.size
            params = letterbox_params(
                frame_w=frame_w, frame_h=frame_h, display_w=display_w, display_h=display_h
            )
            img_arr = np.asarray(img.resize((display_w, display_h)))
        else:
            st.warning("Source media unavailable — bbox click uses synthetic canvas.")
            frame_w, frame_h = 320, 240
            params = letterbox_params(
                frame_w=frame_w, frame_h=frame_h, display_w=display_w, display_h=display_h
            )
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            img_arr = canvas

        fig = go.Figure(go.Image(z=img_arr))
        for b in bboxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            # map frame→display with letterbox
            assert params is not None
            dx1 = params.pad_x + x1 * params.scale
            dy1 = params.pad_y + y1 * params.scale
            dx2 = params.pad_x + x2 * params.scale
            dy2 = params.pad_y + y2 * params.scale
            fig.add_shape(
                type="rect",
                x0=dx1,
                y0=dy1,
                x1=dx2,
                y1=dy2,
                line=dict(color="yellow", width=2),
            )
        fig.update_layout(
            width=display_w,
            height=display_h,
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode=False,
        )
        fig.update_xaxes(visible=False, range=[0, display_w])
        fig.update_yaxes(visible=False, range=[display_h, 0])

        selection = st.plotly_chart(
            fig,
            on_select="rerun",
            selection_mode="points",
            key=f"frame_plot_{event['event_id']}_{frame_index}",
        )
        # Fallback: numeric click entry for deterministic tests / environments
        with st.expander("Manual click coordinates (UI pixels)"):
            cx = st.number_input("ui_x", value=0.0)
            cy = st.number_input("ui_y", value=0.0)
            if st.button("Resolve click"):
                result = resolve_click_to_bbox(
                    ui_x=float(cx), ui_y=float(cy), params=params, bboxes=bboxes
                )
                st.write(result)
                if result.status == "hit" and result.selected is not None:
                    listed = find_candidate_covering_bbox(
                        candidates,
                        frame_index=frame_index,
                        bbox_xyxy=result.selected.bbox_xyxy,
                    )
                    session.selection = {
                        "selected_candidate_id": None if listed is None else listed["candidate_id"],
                        "selected_segment_id": (
                            listed["segment_id"]
                            if listed
                            else result.selected.provenance.get("segment_id")
                            or result.selected.bbox_id
                        ),
                        "selected_raw_track_id": (
                            listed["raw_track_id"]
                            if listed
                            else result.selected.provenance.get("raw_track_id")
                        ),
                        "selected_frame_index": frame_index,
                        "selected_bbox_xyxy": list(result.selected.bbox_xyxy),
                        "direct_bbox_selection": listed is None,
                        "listed_selection": listed is not None,
                        "displayed_rank": None
                        if listed is None
                        else listed.get("appearance_rank"),
                        "displayed_score": None if listed is None else listed.get("S"),
                        "displayed_T_max": None if listed is None else listed.get("T_max"),
                        "displayed_D_max": None if listed is None else listed.get("D_max"),
                        "displayed_model_id": None
                        if listed is None
                        else listed.get("sportsreid_model_id"),
                        "displayed_checkpoint_sha256": None
                        if listed is None
                        else listed.get("sportsreid_checkpoint_sha256"),
                    }
                    if listed is None:
                        st.warning("Direct non-candidate bbox selection")
                    else:
                        st.success(f"Listed candidate selected: {listed['candidate_id']}")

        # Plotly native click (best-effort)
        try:
            points = selection.selection.points if selection and selection.selection else []
            if points:
                pt = points[0]
                ux = float(pt.get("x", 0))
                uy = float(pt.get("y", 0))
                result = resolve_click_to_bbox(
                    ui_x=ux, ui_y=uy, params=params, bboxes=bboxes
                )
                st.write({"plotly_click": {"x": ux, "y": uy}, "resolution": result.status})
        except Exception:  # noqa: BLE001
            pass

        st.write("Current selection:", session.selection or None)

        st.subheader("Candidates")
        for cand in candidates:
            with st.expander(
                f"{cand['candidate_id']} · rank={cand.get('appearance_rank')} · "
                f"seg={cand['segment_id']}"
            ):
                st.json(sportsreid_display_fields(cand))
                st.write(
                    {
                        "raw_track_id": cand["raw_track_id"],
                        "frames": [cand["start_frame"], cand["middle_frame"], cand["end_frame"]],
                        "quality": cand.get("quality"),
                        "contamination": cand.get("contamination"),
                    }
                )
                if st.button("Select listed", key=f"sel_{cand['candidate_id']}"):
                    refs = cand.get("bbox_references") or []
                    ref = refs[0] if refs else None
                    session.selection = {
                        "selected_candidate_id": cand["candidate_id"],
                        "selected_segment_id": cand["segment_id"],
                        "selected_raw_track_id": cand["raw_track_id"],
                        "selected_frame_index": None if ref is None else ref["frame_index"],
                        "selected_bbox_xyxy": None if ref is None else list(ref["bbox_xyxy"]),
                        "direct_bbox_selection": False,
                        "listed_selection": True,
                        "displayed_rank": cand.get("appearance_rank"),
                        "displayed_score": cand.get("S"),
                        "displayed_T_max": cand.get("T_max"),
                        "displayed_D_max": cand.get("D_max"),
                        "displayed_model_id": cand.get("sportsreid_model_id"),
                        "displayed_checkpoint_sha256": cand.get(
                            "sportsreid_checkpoint_sha256"
                        ),
                    }

        reviewer = st.text_input("Reviewer", value="hil_b_reviewer")
        comment = st.text_area("Comment", value="")
        confidence = st.selectbox("Confidence", ["unknown", "probable", "confirmed"])
        st.checkbox("training_use_approved", value=False, disabled=True)
        st.checkbox("gallery_use_approved", value=False, disabled=True)
        st.caption("Gallery/training approval defaults false and are disabled in MVP.")

        action_labels = [
            "Confirm Target",
            "Reject Candidate",
            "None of These",
            "Unknown",
            "Invalid Segment",
            "Defer",
            "Undo Last Decision",
            "Correct Previous Decision",
            "Revoke Decision",
        ]
        chosen = st.selectbox("Action", action_labels)
        if chosen == "Confirm Target":
            st.write("Confirmation preview")
            from football_analytics.reid.hil_ui.actions import preview_confirmation

            st.json(
                preview_confirmation(
                    action=DecisionAction.CONFIRM_TARGET,
                    selection=session.selection,
                    reviewer=reviewer,
                    comment=comment,
                    confidence=confidence,
                ).to_dict()
            )

        if st.button("Submit decision"):
            try:
                if chosen == "Undo Last Decision":
                    result = undo_last_decision(
                        session.decision_log,
                        event=event,
                        candidate_manifest=man,
                        reviewer=reviewer,
                        comment=comment or "undo",
                    )
                elif chosen in {"Revoke Decision", "Correct Previous Decision"}:
                    act = map_ui_action(chosen)
                    result = correct_or_revoke(
                        session.decision_log,
                        event=event,
                        candidate_manifest=man,
                        reviewer=reviewer,
                        action=act,
                        comment=comment,
                        selection=session.selection,
                    )
                else:
                    act = map_ui_action(chosen)
                    result = submit_decision(
                        session.decision_log,
                        event=event,
                        candidate_manifest=man,
                        action=act,
                        reviewer=reviewer,
                        selection=session.selection,
                        comment=comment,
                        confidence=confidence,
                        training_use_approved=False,
                        gallery_use_approved=False,
                    )
                st.success("Appended to decision log")
                st.json(
                    {
                        "decision_id": result["decision"]["decision_id"],
                        "action": result["decision"]["action"],
                        "revision": result["decision"]["revision"],
                        "log_sha256": result["log_integrity"]["sha256"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with tab_hist:
        event = session.current_event
        hist = history_for_event(session.decision_log, event["event_id"])
        st.write("Append-only raw records (edit/delete UI not provided)")
        st.json(hist)


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
