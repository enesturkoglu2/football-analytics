"""Streamlit entrypoint for offline HIL target recovery review (local loopback only)."""

from __future__ import annotations

import json
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
from football_analytics.reid.hil_ui.compat import streamlit_image  # noqa: E402
from football_analytics.reid.hil_ui.dense_observations import (  # noqa: E402
    attach_candidate_ids,
    build_dense_observations_from_mapping,
    load_mapping_jsonl,
)
from football_analytics.reid.hil_ui.gallery_quality import audit_gallery_candidates  # noqa: E402
from football_analytics.reid.hil_ui.gallery_view import (  # noqa: E402
    validate_gallery_crop_for_display,
)
from football_analytics.reid.hil_ui.interactive_video_component import (  # noqa: E402
    interactive_video_review,
)
from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: E402
    ensure_interactive_review_proxy,
)
from football_analytics.reid.hil_ui.media import decode_frame_to_cache  # noqa: E402
from football_analytics.reid.hil_ui.observations import load_observation_lookup  # noqa: E402
from football_analytics.reid.hil_c2.product_package import _segment_id  # noqa: E402
from football_analytics.reid.multi_event_hil.review_clips import (  # noqa: E402
    bboxes_from_candidates_for_window,
    extract_event_window_clip,
)
from football_analytics.reid.hil.common import sha256_file  # noqa: E402
from football_analytics.reid.hil.resolve import resolve_effective_decisions  # noqa: E402
from football_analytics.reid.hil.timeline.approvals import (  # noqa: E402
    ApprovalLog,
    assert_decision_approvable,
    build_approval_record,
)
from football_analytics.reid.multi_event_hil.gallery_approvals import (  # noqa: E402
    GalleryApprovalLog,
    build_gallery_approval,
    resolve_active_gallery_approvals,
)
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
    package_mode = str((session.package.get("provenance") or {}).get("package_mode") or "")
    is_product = package_mode == "product" or str(session.package.get("run_id", "")).startswith(
        ("hil_c2_product", "mehil_run")
    )
    if is_product:
        st.info(
            "PRODUCT package · decisions are NOT timeline-eligible until explicit "
            "Timeline Approval on the Approvals tab."
        )

    tab_live, tab_pkg, tab_queue, tab_review, tab_hist, tab_appr, tab_gal = st.tabs(
        [
            "Interactive Video",
            "Session / Package",
            "Recovery Queue",
            "Frame Fallback / Debug",
            "Decision History",
            "Timeline Approvals",
            "Match Gallery",
        ]
    )

    with tab_live:
        st.subheader("True interactive video tracking review")
        st.caption(
            "HTML5 video + synchronized canvas overlay. Click a moving bbox to select a tracklet. "
            "Playback does not rerun detection/tracking."
        )
        event = session.current_event
        st.write(
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "window": [
                    event["review_window_start_frame"],
                    event["review_window_end_frame"],
                ],
            }
        )
        nav_a, nav_b = st.columns(2)
        if nav_a.button("Previous Event", key="live_prev"):
            session.previous_event()
            st.session_state.selection = {}
            st.rerun()
        if nav_b.button("Next Event", key="live_next"):
            session.next_event()
            st.session_state.selection = {}
            st.rerun()

        man = session.current_manifest()
        candidates = sort_candidates_for_display(man["candidates"]) if man else []

        # Dense observations from external mapping (read-only)
        mapping_path = None
        for root in session.package.get("read_only_source_roots_resolved") or []:
            cand = (
                Path(root)
                / "inventory"
                / "target_001_external_track_candidate_mapping.jsonl"
            )
            if cand.is_file():
                mapping_path = cand
                break
        dens = {}
        if mapping_path is not None:
            codes = sorted(
                {
                    str((c.get("metadata") or {}).get("external_candidate_code") or "")
                    for c in candidates
                }
                | {
                    str(c["segment_id"]).replace("EXT_SEG_", "EXT_")
                    for c in candidates
                    if str(c.get("segment_id", "")).startswith("EXT_SEG_")
                }
            )
            codes = [c if c.startswith("EXT_") else c for c in codes if c]
            # normalize EXT_SEG_004 → EXT_004
            norm_codes = []
            for c in codes:
                if c.startswith("EXT_SEG_"):
                    norm_codes.append("EXT_" + c.replace("EXT_SEG_", ""))
                elif c.startswith("EXT_"):
                    norm_codes.append(c)
            dens = build_dense_observations_from_mapping(
                load_mapping_jsonl(mapping_path),
                codes=sorted(set(norm_codes)),
                segment_id_fn=_segment_id,
            )
            dens = attach_candidate_ids(dens, candidates)

        markers = []
        for ev in session.events:
            markers.append(
                {
                    "event_id": ev["event_id"],
                    "label": f"{ev['event_type'].split('_')[-1]} @ {ev['review_window_start_frame']}",
                    "frame_index": int(ev["review_window_start_frame"]),
                }
            )

        sel = st.session_state.selection or {}
        track_end = None
        if sel.get("selected_segment_id") and dens:
            # infer end from last frame containing that segment
            last = None
            for fi_s, rows in dens.items():
                if any(r.get("segment_id") == sel.get("selected_segment_id") for r in rows):
                    last = int(fi_s)
            track_end = last

        proxy_info = None
        if session.package.get("source_video_available"):
            proxy_path = (
                Path(session.package["writable_session_root_resolved"])
                / "interactive_review"
                / "source_proxy_960.mp4"
            )
            try:
                proxy_info = ensure_interactive_review_proxy(
                    source_video=Path(session.package["source_video_resolved"]),
                    source_video_sha256=session.package["source_video_sha256"],
                    output_path=proxy_path,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Interactive video proxy failed: {exc}")

        if proxy_info:
            click = interactive_video_review(
                video_data_url=proxy_info["data_url"],
                observations_by_frame=dens,
                markers=markers,
                selected_raw_track_id=sel.get("selected_raw_track_id"),
                selected_segment_id=sel.get("selected_segment_id"),
                track_end_frame=track_end,
                fps=30.0,
                video_width=1332,
                video_height=746,
                key=f"interactive_{event['event_id']}",
            )
            if isinstance(click, dict) and click.get("type") == "bbox_click":
                st.session_state.selection = {
                    "selected_candidate_id": click.get("candidate_id"),
                    "selected_segment_id": click.get("segment_id"),
                    "selected_raw_track_id": click.get("raw_track_id"),
                    "selected_frame_index": click.get("frame_index"),
                    "selected_bbox_xyxy": click.get("bbox_xyxy"),
                    "direct_bbox_selection": click.get("candidate_id") is None,
                    "listed_selection": click.get("candidate_id") is not None,
                    "displayed_rank": None,
                    "displayed_score": None,
                    "displayed_T_max": None,
                    "displayed_D_max": None,
                    "displayed_model_id": None,
                    "displayed_checkpoint_sha256": None,
                    "source": "interactive_video_bbox_click",
                }
                st.success(
                    f"Video click selection · {click.get('segment_id')} / "
                    f"track {click.get('raw_track_id')} @ frame {click.get('frame_index')}"
                )
            elif isinstance(click, dict) and click.get("type") == "marker_seek":
                for i, ev in enumerate(session.events):
                    if ev["event_id"] == click.get("event_id"):
                        session.set_event_index(i)
                        break

            card = selection_summary_card(st.session_state.selection or None)
            if card:
                st.success("Selected target card")
                st.write(card)
                st.write(
                    {
                        "track_start_end_inferred": [
                            None,
                            track_end,
                        ],
                        "proxy_sha256": proxy_info["proxy_sha256"],
                    }
                )

            reviewer = st.text_input("Reviewer", value="hil_b_reviewer", key="live_reviewer")
            comment = st.text_area("Comment", value="", key="live_comment")
            confidence = st.selectbox(
                "Confidence", ["unknown", "probable", "confirmed"], key="live_conf"
            )
            if st.button("Submit CONFIRM_TARGET from video selection", key="live_confirm"):
                try:
                    if not st.session_state.selection:
                        raise RuntimeError("no video selection")
                    result = submit_decision(
                        session.decision_log,
                        event=event,
                        candidate_manifest=man,
                        action=DecisionAction.CONFIRM_TARGET,
                        reviewer=reviewer,
                        selection=st.session_state.selection,
                        comment=comment,
                        confidence=confidence,
                        training_use_approved=False,
                        gallery_use_approved=False,
                    )
                    st.success(SUBMIT_SUCCESS_MESSAGE)
                    st.json(
                        {
                            "decision_id": result["decision"]["decision_id"],
                            "log_sha256": result["log_integrity"]["sha256"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
            st.caption("Timeline Approval remains on the Timeline Approvals tab.")
        else:
            st.warning("Source video unavailable for interactive review.")

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
        st.info(
            "Video playback is for review. Clicking and confirming links existing "
            "tracklets; it does not rerun tracking."
        )
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

        # --- Continuous video review (full source + event window) ---
        if session.package.get("source_video_available"):
            video_path = Path(session.package["source_video_resolved"])
            st.markdown("### Full source preview")
            st.caption(
                f"Read-only source · SHA `{session.package['source_video_sha256'][:16]}…` · "
                f"current event window frames {start}–{end}"
            )
            try:
                st.video(str(video_path))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Full source playback failed: {exc}")

            man_preview = session.current_manifest()
            cands_preview = list((man_preview or {}).get("candidates") or [])
            clip_root = (
                Path(session.package["writable_session_root_resolved"]) / "review_clips"
            )
            try:
                bbox_map = bboxes_from_candidates_for_window(
                    cands_preview, start_frame=start, end_frame=end
                )
                clip_man = extract_event_window_clip(
                    source_video=video_path,
                    source_video_sha256=session.package["source_video_sha256"],
                    event=event,
                    output_dir=clip_root,
                    candidate_bboxes_by_frame=bbox_map,
                )
                st.markdown("### Event-window clip")
                st.caption(
                    f"Contract window {start}–{end} · no invented padding · "
                    f"overlay `{Path(clip_man['overlay_clip_path']).name}`"
                )
                st.video(clip_man["overlay_clip_path"])
                with st.expander("Clip technical details"):
                    st.json(
                        {
                            "copy_clip_sha256": clip_man["copy_clip_sha256"],
                            "overlay_clip_sha256": clip_man["overlay_clip_sha256"],
                            "source_video_sha256": clip_man["source_video_sha256"],
                            "duration_seconds": clip_man["duration_seconds"],
                            "padding_invented": clip_man["padding_invented"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Event-window clip failed: {exc}")

        frame_index = st.slider(
            "Frame (bbox selection)", min_value=start, max_value=max(start, end), value=start
        )
        st.write(f"Current frame index: {frame_index}")

        man = session.current_manifest()
        candidates = sort_candidates_for_display(man["candidates"]) if man else []
        gallery_ready = bool(
            (session.package.get("provenance") or {}).get("match_specific_gallery_ready")
        )
        if not gallery_ready and any(
            c.get("appearance_rank") is not None for c in candidates
        ):
            st.warning(
                "Appearance ranks present without match-specific gallery flag; "
                "treat ranks as helper-only."
            )
        if not any(c.get("appearance_rank") is not None for c in candidates):
            st.info("Match-specific appearance gallery is not ready (no helper ranks).")
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

    with tab_appr:
        st.subheader("Product timeline approval")
        if not is_product:
            st.warning(
                "This package is not product-mode. Timeline approvals are disabled "
                "(fixture/acceptance/dev packages cannot mint product eligibility)."
            )
        else:
            approval_path = (session.package.get("provenance") or {}).get("approval_log_path")
            if not approval_path:
                st.error("approval_log_path missing from package provenance")
            else:
                st.caption(
                    "A CONFIRM_TARGET decision alone does not enter the product timeline. "
                    "Approve only after reviewing the summary below. Gaps stay unresolved."
                )
                decisions = session.decision_log.read_raw()
                by_id = {d["decision_id"]: d for d in decisions}
                effective = resolve_effective_decisions(decisions)
                confirm_tips = []
                for tip in effective.values():
                    if tip.get("action") != "CONFIRM_TARGET":
                        continue
                    row = by_id.get(tip["effective_decision_id"])
                    if row is not None:
                        confirm_tips.append(row)
                if not confirm_tips:
                    st.info("No effective CONFIRM_TARGET decisions yet.")
                else:
                    options = {
                        f"{d['decision_id']} · {d.get('selected_segment_id')} · "
                        f"frame {d.get('selected_frame_index')}": d
                        for d in confirm_tips
                    }
                    chosen_label = st.selectbox("Decision to approve", list(options.keys()))
                    chosen_dec = options[chosen_label]
                    st.write(
                        {
                            "decision_id": chosen_dec["decision_id"],
                            "target_id": chosen_dec.get("target_id"),
                            "segment_id": chosen_dec.get("selected_segment_id"),
                            "raw_track_id": chosen_dec.get("selected_raw_track_id"),
                            "video_id": chosen_dec.get("video_id"),
                            "frame": chosen_dec.get("selected_frame_index"),
                            "enters_product_timeline": True,
                            "usable_for_heatmap_distance": True,
                            "unresolved_gaps_auto_filled": False,
                        }
                    )
                    ack = st.checkbox(
                        "I explicitly approve this decision for product_target_timeline",
                        value=False,
                    )
                    appr_comment = st.text_input("Approval comment", value="")
                    if st.button("Append timeline approval", disabled=not ack):
                        try:
                            log_path = session.package["decision_log_resolved"]
                            log_sha = sha256_file(Path(log_path))
                            assert_decision_approvable(
                                chosen_dec,
                                log_path=log_path,
                                log_sha256=log_sha,
                                review_package_mode="product",
                                product_package_id=session.package["package_id"],
                                expected_target_id=session.package["target_id"],
                                expected_video_sha256=session.package["source_video_sha256"],
                            )
                            approval_id = f"appr_{chosen_dec['decision_id']}_{len(ApprovalLog(approval_path).read_raw())+1:04d}"
                            record = build_approval_record(
                                approval_id=approval_id,
                                decision=chosen_dec,
                                product_package_id=session.package["package_id"],
                                decision_log_path=log_path,
                                decision_log_sha256_at_approval=log_sha,
                                comment=appr_comment,
                                candidate_manifest_sha256=(
                                    list(session.package.get("candidate_manifest_sha256", {}).values())
                                    or [None]
                                )[0],
                                segment_manifest_sha256=(
                                    session.package.get("provenance") or {}
                                ).get("segment_inventory_sha256"),
                                provenance={
                                    "explicit_user_approval": True,
                                    "package_mode": "product",
                                },
                            )
                            ApprovalLog(approval_path).append(record)
                            st.success(f"Approval appended: {approval_id}")
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))
                st.write("Approval log (append-only)")
                st.json(ApprovalLog(approval_path).read_raw())

    with tab_gal:
        st.subheader("Match-specific gallery crop approvals")
        st.info(
            "These are multiple views of the same enrolled target tracklet. Approve "
            "only clear, uncontaminated views for this match-specific appearance gallery."
        )
        st.caption(
            "Development/old target gallery is forbidden. Approving a crop does not "
            "auto-expand training use. Timeline eligibility still requires Timeline Approvals. "
            "Approval is disabled unless the crop image is visible and SHA-verified."
        )
        gal_log_path = (session.package.get("provenance") or {}).get(
            "gallery_approval_log_path"
        )
        crop_man_path = (
            session.package_path.parent
            / "gallery_crop_candidates"
            / "enrollment_crop_candidates.json"
        )
        if not is_product:
            st.warning("Gallery approvals enabled only for product-mode packages.")
        elif not gal_log_path:
            st.info("No gallery_approval_log_path in package provenance.")
        elif not crop_man_path.is_file():
            st.info(
                f"Crop candidates not found yet: {crop_man_path}. "
                "Run multi-event package builder first."
            )
        else:
            crop_man = json.loads(crop_man_path.read_text(encoding="utf-8"))
            gal_log = GalleryApprovalLog(gal_log_path)
            active = resolve_active_gallery_approvals(gal_log.read_raw())
            quality = audit_gallery_candidates(
                crop_man.get("candidates") or [],
                source_frame_size=(1332, 746),
            )
            st.warning(quality["user_message"])
            segs = {c.get("segment_id") for c in crop_man.get("candidates") or []}
            tracks = {str(c.get("raw_track_id")) for c in crop_man.get("candidates") or []}
            st.write(
                {
                    "candidate_count": crop_man.get("candidate_count"),
                    "usable_gallery_candidates": quality["usable_count"],
                    "same_segment_ids": sorted(segs),
                    "same_raw_tracks": sorted(tracks),
                    "active_approvals": len(active),
                    "automatic_gallery_expansion": False,
                    "note": "Not five different players — views of one enrolled tracklet.",
                }
            )
            match_id = (session.package.get("provenance") or {}).get("match_id") or "unknown"
            analysis_run_id = session.package.get("run_id")
            st.text_input("Gallery reviewer", key="gal_reviewer", value="hil_gallery_reviewer")
            audits_by_id = {a.get("crop_id"): a for a in quality["audits"]}
            for crop in crop_man.get("candidates") or []:
                validation = audits_by_id.get(crop["crop_id"]) or validate_gallery_crop_for_display(
                    crop
                )
                cols = st.columns([2, 3, 1])
                with cols[0]:
                    if validation.get("visible"):
                        from PIL import Image

                        streamlit_image(
                            cols[0],
                            Image.open(crop["crop_path"]),
                            caption=f"{crop['crop_id']} · {validation.get('quality_class')}",
                            use_column_width=True,
                        )
                    else:
                        cols[0].error(validation.get("error") or "image not visible")
                cols[1].write(
                    {
                        "crop_id": crop["crop_id"],
                        "frame": crop["frame_index"],
                        "view_hint": crop.get("view_hint"),
                        "segment_id": crop["segment_id"],
                        "raw_track_id": crop["raw_track_id"],
                        "image_dimensions": [
                            validation.get("width"),
                            validation.get("height"),
                        ],
                        "player_pixel_height": validation.get("player_pixel_height"),
                        "blur_laplacian_variance": validation.get("blur_laplacian_variance"),
                        "clipping": validation.get("clipping"),
                        "quality_class": validation.get("quality_class"),
                        "approved": crop["crop_id"] in active,
                        "approval_enabled": bool(validation.get("approval_enabled")),
                    }
                )
                already = crop["crop_id"] in active
                approve_disabled = (not validation.get("approval_enabled")) or already
                if cols[2].button(
                    "Approve gallery member",
                    key=f"gal_{crop['crop_id']}",
                    disabled=approve_disabled,
                ):
                    try:
                        if not validation.get("approval_enabled"):
                            raise RuntimeError(
                                "approval blocked: crop failed gallery quality gate"
                            )
                        record = build_gallery_approval(
                            approval_id=(
                                f"gal_{crop['crop_id']}_"
                                f"{len(gal_log.read_raw()) + 1:04d}"
                            ),
                            crop=crop,
                            match_id=str(match_id),
                            analysis_run_id=str(analysis_run_id),
                            target_id=session.package["target_id"],
                            product_package_id=session.package["package_id"],
                            reviewer=st.session_state.get(
                                "gal_reviewer", "hil_gallery_reviewer"
                            ),
                        )
                        gal_log.append(record)
                        st.success(f"Gallery approval appended for {crop['crop_id']}")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
                if already and cols[2].button(
                    "Revoke gallery approval",
                    key=f"gal_revoke_{crop['crop_id']}",
                ):
                    try:
                        tip = active[crop["crop_id"]]
                        record = build_gallery_approval(
                            approval_id=(
                                f"gal_{crop['crop_id']}_"
                                f"{len(gal_log.read_raw()) + 1:04d}"
                            ),
                            crop=crop,
                            match_id=str(match_id),
                            analysis_run_id=str(analysis_run_id),
                            target_id=session.package["target_id"],
                            product_package_id=session.package["package_id"],
                            reviewer=st.session_state.get(
                                "gal_reviewer", "hil_gallery_reviewer"
                            ),
                            approval_status="revoked",
                            supersedes_approval_id=tip["approval_id"],
                            comment="revoked_via_ui",
                        )
                        gal_log.append(record)
                        st.success(f"Gallery approval revoked for {crop['crop_id']}")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
            if quality["usable_count"] == 0:
                st.info(
                    "No USABLE_GALLERY_CANDIDATE crops. Gallery member count stays 0; "
                    "embedding will not run; HIL workflow can continue without helper ranking."
                )
            st.write("Gallery approval log (append-only)")
            st.json(gal_log.read_raw())


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
