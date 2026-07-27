"""Streamlit entrypoint for offline HIL target recovery review (local loopback only)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Heavy model imports are intentionally absent.

HELPER_WARNING = "SportsReID is a helper ranker; it does not confirm identity."


def _ensure_src_path() -> None:
    import sys

    src = Path(__file__).resolve().parents[3]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_path()

from football_analytics.reid.hil.decisions import DecisionAction  # noqa: E402
from football_analytics.reid.hil_ui.actions import map_ui_action, preview_confirmation  # noqa: E402
from football_analytics.reid.hil_ui.candidates_view import sort_candidates_for_display  # noqa: E402
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
from football_analytics.reid.hil_ui.observations import load_observation_lookup  # noqa: E402
from football_analytics.reid.hil_ui.session import open_review_session  # noqa: E402
from football_analytics.reid.hil_ui.visualization import (  # noqa: E402
    SPARSE_PACKAGE_NOTICE,
    SUBMIT_SUCCESS_MESSAGE,
    TRACKING_NOT_RUN_NOTICE,
    action_requires_selection,
    appearance_rank_label,
    build_selection_from_bbox_hit,
    confirmation_user_summary,
    is_sparse_observations,
    observation_frames,
    selection_summary_card,
    selection_visibility,
)


def _package_path_from_env() -> Path:
    raw = os.environ.get("HIL_REVIEW_PACKAGE")
    if not raw:
        raise RuntimeError(
            "HIL_REVIEW_PACKAGE env var is required (set by launch script)"
        )
    return Path(raw).expanduser().resolve()


def _frame_bboxes_for_event(
    session,
    frame_index: int,
    observation_lookup: dict[tuple[str, int], list[float]],
) -> list[dict[str, Any]]:
    man = session.current_manifest()
    rows: list[dict[str, Any]] = []
    if not man:
        return rows
    seen_segments: set[str] = set()
    for cand in man.get("candidates", []):
        seg = cand["segment_id"]
        bbox = None
        for ref in cand.get("bbox_references") or []:
            if int(ref["frame_index"]) == int(frame_index):
                bbox = list(ref["bbox_xyxy"])
                break
        if bbox is None:
            hit = observation_lookup.get((seg, int(frame_index)))
            if hit is not None:
                bbox = list(hit)
        if bbox is None:
            continue
        seen_segments.add(seg)
        rows.append(
            {
                "bbox_id": f"{cand['candidate_id']}:{frame_index}",
                "candidate_id": cand["candidate_id"],
                "segment_id": seg,
                "raw_track_id": cand["raw_track_id"],
                "bbox_xyxy": bbox,
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
    extras = (session.package.get("provenance") or {}).get("extra_frame_bboxes") or {}
    for item in extras.get(str(frame_index), []):
        rows.append(dict(item))
    return rows


def _draw_overlay_image(
    *,
    base_rgb,
    bboxes: list[dict[str, Any]],
    selected_bbox: list[float] | None,
    display_w: int,
):
    import cv2
    import numpy as np
    from PIL import Image

    frame = np.asarray(base_rgb).copy()
    for b in bboxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox_xyxy"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
    if selected_bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in selected_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 64, 0), 4)
        cv2.putText(
            frame,
            "SELECTED TARGET CANDIDATE",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 64, 0),
            2,
            cv2.LINE_AA,
        )
    img = Image.fromarray(frame)
    frame_w, frame_h = img.size
    scale = display_w / float(frame_w)
    display_h = max(1, int(round(frame_h * scale)))
    shown = img.resize((display_w, display_h))
    return shown, frame_w, frame_h, display_w, display_h


def run_app() -> None:
    import streamlit as st
    from PIL import Image
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.set_page_config(page_title="HIL Offline Review", layout="wide")
    st.title("Target Recovery Offline Review")
    st.caption("Local loopback only · no inference · append-only decisions")
    st.warning(HELPER_WARNING)

    package_path = _package_path_from_env()
    if "session" not in st.session_state:
        st.session_state.session = open_review_session(package_path)
    if "selection" not in st.session_state:
        st.session_state.selection = {}
    session = st.session_state.session

    tab_pkg, tab_queue, tab_review, tab_hist = st.tabs(
        ["Session / Package", "Recovery Queue", "Recovery Review", "Decision History"]
    )

    with tab_pkg:
        summary = session.package_summary()
        st.write(
            {
                "package_id": summary["package_id"],
                "target_id": summary["target_id"],
                "event_count": summary["event_count"],
                "decision_log_path": summary["decision_log_path"],
            }
        )
        with st.expander("Advanced technical details"):
            st.json(summary)

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
                        st.session_state.selection = {}
                        break
                st.rerun()

    with tab_review:
        event = session.current_event
        st.info(TRACKING_NOT_RUN_NOTICE)
        st.subheader(f"{event['event_id']} · {event['event_type']}")
        nav1, nav2 = st.columns(2)
        if nav1.button("Previous Event"):
            session.previous_event()
            st.session_state.selection = {}
            st.rerun()
        if nav2.button("Next Event"):
            session.next_event()
            st.session_state.selection = {}
            st.rerun()

        start = int(event["review_window_start_frame"])
        end = int(event["review_window_end_frame"])
        frame_index = st.slider(
            "Frame", min_value=start, max_value=max(start, end), value=start
        )
        st.write(f"Current frame index: {frame_index}")

        man = session.current_manifest()
        candidates = sort_candidates_for_display(man["candidates"]) if man else []
        if is_sparse_observations(
            candidates, review_window_start=start, review_window_end=end
        ):
            st.warning(SPARSE_PACKAGE_NOTICE)

        seg_ids = {c["segment_id"] for c in candidates}
        obs_path = (session.package.get("provenance") or {}).get("observation_index_path")
        observation_lookup = load_observation_lookup(
            obs_path,
            segment_ids=seg_ids,
            frame_min=start,
            frame_max=end,
        )

        bboxes = _frame_bboxes_for_event(session, frame_index, observation_lookup)
        vis = selection_visibility(
            selection=st.session_state.selection or None,
            frame_index=frame_index,
            candidates=candidates,
            observation_lookup=observation_lookup,
        )
        display_w = 960

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
                overlay_bboxes=None,
            )
            base = Image.open(decoded["path"]).convert("RGB")
        else:
            st.warning("Source media unavailable.")
            base = Image.new("RGB", (320, 240), (30, 30, 30))

        shown, frame_w, frame_h, disp_w, disp_h = _draw_overlay_image(
            base_rgb=base,
            bboxes=bboxes,
            selected_bbox=vis.get("bbox_xyxy"),
            display_w=display_w,
        )
        # Uniform scale (no letterbox pad): click coords map by simple ratio.
        params = letterbox_params(
            frame_w=frame_w,
            frame_h=frame_h,
            display_w=disp_w,
            display_h=disp_h,
        )

        st.caption(
            "Click a yellow box on the image to select a player "
            "(direct click). Overlay shapes are burned into the image."
        )
        click = streamlit_image_coordinates(
            shown,
            key=f"img_click_{event['event_id']}_{frame_index}",
            width=disp_w,
        )
        if click and "x" in click and "y" in click:
            result = resolve_click_to_bbox(
                ui_x=float(click["x"]),
                ui_y=float(click["y"]),
                params=params,
                bboxes=bboxes,
            )
            if result.status == "hit" and result.selected is not None:
                meta = {}
                for b in bboxes:
                    if b["bbox_id"] == result.selected.bbox_id:
                        meta = {
                            "segment_id": b.get("segment_id"),
                            "raw_track_id": b.get("raw_track_id"),
                            "bbox_id": b.get("bbox_id"),
                            **dict(b.get("provenance") or {}),
                        }
                        break
                st.session_state.selection = build_selection_from_bbox_hit(
                    frame_index=frame_index,
                    bbox_xyxy=result.selected.bbox_xyxy,
                    candidates=candidates,
                    hit_meta=meta,
                )
                st.success(
                    "Image click selection applied"
                    + (
                        " (direct)"
                        if st.session_state.selection.get("direct_bbox_selection")
                        else " (listed)"
                    )
                )
            elif result.status == "miss":
                st.caption("Click did not land inside a bbox.")

        st.subheader("Players visible in this frame")
        if not bboxes:
            st.write("No bbox observations on this frame.")
        for b in bboxes:
            label = (
                f"{b.get('candidate_id') or b['bbox_id']} · "
                f"{b.get('segment_id')} · track {b.get('raw_track_id')}"
            )
            if st.button(f"Select this player · {label}", key=f"pick_{b['bbox_id']}"):
                st.session_state.selection = build_selection_from_bbox_hit(
                    frame_index=frame_index,
                    bbox_xyxy=b["bbox_xyxy"],
                    candidates=candidates,
                    hit_meta={
                        "segment_id": b.get("segment_id"),
                        "raw_track_id": b.get("raw_track_id"),
                        "bbox_id": b.get("bbox_id"),
                        **dict(b.get("provenance") or {}),
                    },
                )
                st.rerun()

        card = selection_summary_card(st.session_state.selection or None)
        if card:
            st.success("Current selection")
            st.write(card)
            if vis.get("visible"):
                st.markdown(f"**{vis['label']}** (highlighted on frame {frame_index})")
            else:
                st.warning(vis.get("message"))
        else:
            st.info("No selection yet.")

        st.subheader("Candidates")
        st.caption(f"Showing all {len(candidates)} candidates (no top-k hiding).")
        for cand in candidates:
            title = (
                f"{cand['candidate_id']} · {appearance_rank_label(cand)} · "
                f"{cand['segment_id']}"
            )
            with st.expander(title):
                frames = observation_frames(cand)
                st.write(
                    {
                        "segment_id": cand["segment_id"],
                        "raw_track_id": cand["raw_track_id"],
                        "start_middle_end": [
                            cand["start_frame"],
                            cand["middle_frame"],
                            cand["end_frame"],
                        ],
                        "observation_frames_in_manifest": frames,
                        "frame_availability": (
                            "sparse" if len(frames) <= 2 else "dense_in_manifest"
                        ),
                        "appearance": appearance_rank_label(cand),
                    }
                )
                selected = (
                    st.session_state.selection or {}
                ).get("selected_candidate_id") == cand["candidate_id"]
                if selected:
                    st.markdown("**Selected**")
                if st.button(
                    "Select this candidate", key=f"sel_{cand['candidate_id']}"
                ):
                    refs = cand.get("bbox_references") or []
                    ref = refs[0] if refs else None
                    st.session_state.selection = {
                        "selected_candidate_id": cand["candidate_id"],
                        "selected_segment_id": cand["segment_id"],
                        "selected_raw_track_id": cand["raw_track_id"],
                        "selected_frame_index": None
                        if ref is None
                        else ref["frame_index"],
                        "selected_bbox_xyxy": None
                        if ref is None
                        else list(ref["bbox_xyxy"]),
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
                    st.rerun()

        with st.expander("Advanced / debug: manual pixel coordinates"):
            cx = st.number_input("ui_x", value=0.0)
            cy = st.number_input("ui_y", value=0.0)
            if st.button("Resolve debug click"):
                result = resolve_click_to_bbox(
                    ui_x=float(cx), ui_y=float(cy), params=params, bboxes=bboxes
                )
                st.write(
                    {
                        "status": result.status,
                        "selected": None
                        if result.selected is None
                        else result.selected.bbox_id,
                    }
                )

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
        needs_sel = action_requires_selection(chosen)
        has_sel = bool(st.session_state.selection)

        if chosen == "Confirm Target":
            st.subheader("Confirmation")
            st.write(
                confirmation_user_summary(
                    action="CONFIRM_TARGET",
                    selection=st.session_state.selection or None,
                    confidence=confidence,
                )
            )
            with st.expander("Advanced technical details"):
                st.json(
                    preview_confirmation(
                        action=DecisionAction.CONFIRM_TARGET,
                        selection=st.session_state.selection,
                        reviewer=reviewer,
                        comment=comment,
                        confidence=confidence,
                    ).to_dict()
                )

        submit_disabled = needs_sel and not has_sel
        if submit_disabled:
            st.warning("Confirm/Reject requires a selection.")

        if st.button("Submit decision", disabled=submit_disabled):
            try:
                sel = st.session_state.selection or {}
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
                        selection=sel,
                    )
                else:
                    act = map_ui_action(chosen)
                    result = submit_decision(
                        session.decision_log,
                        event=event,
                        candidate_manifest=man,
                        action=act,
                        reviewer=reviewer,
                        selection=sel,
                        comment=comment,
                        confidence=confidence,
                        training_use_approved=False,
                        gallery_use_approved=False,
                    )
                st.success(SUBMIT_SUCCESS_MESSAGE)
                with st.expander("Advanced technical details"):
                    st.json(
                        {
                            "decision_id": result["decision"]["decision_id"],
                            "action": result["decision"]["action"],
                            "revision": result["decision"]["revision"],
                            "direct_bbox_selection": result["decision"].get(
                                "direct_bbox_selection"
                            ),
                            "log_sha256": result["log_integrity"]["sha256"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with tab_hist:
        event = session.current_event
        hist = history_for_event(session.decision_log, event["event_id"])
        st.write("Append-only history (edit/delete yok). Revoke/Correct actions above.")
        st.write(
            {
                "review_state": hist["review_state"],
                "effective": hist["effective_active_decision"],
                "chain_len": len(hist["supersedes_chain"]),
            }
        )
        with st.expander("Advanced technical details"):
            st.json(hist)


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
