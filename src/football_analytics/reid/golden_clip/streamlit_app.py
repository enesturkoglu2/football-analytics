"""Streamlit Target Ground Truth annotation UI (isolated from product decisions)."""

from __future__ import annotations

import json
import os
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
from football_analytics.reid.golden_clip.intervals import (  # noqa: E402
    merge_intervals,
    proposal_from_track_span,
    rebuild_gt_from_events,
    split_interval_at_frame,
    track_span_from_observations,
)
from football_analytics.reid.golden_clip.schema import (  # noqa: E402
    active_intervals,
    build_annotation_event,
    build_annotation_interval,
    empty_ground_truth,
    new_session_id,
)
from football_analytics.reid.golden_clip.validate import validate_ground_truth  # noqa: E402
from football_analytics.reid.hil_ui.interactive_video_component import (  # noqa: E402
    interactive_video_review,
)
from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: E402
    ensure_interactive_review_proxy,
)
from football_analytics.reid.short_video.dense_timeline import (  # noqa: E402
    load_dense_timeline,
    observations_for_component,
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


def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Target Ground Truth", layout="wide")
    st.title("Target Ground Truth")
    st.caption(
        "İnsan kimlik ground truth annotation · product decision/approval’dan izole · "
        f"ui_profile={UI_PROFILE}"
    )
    st.info(
        "Bu ekran product Timeline Approval değildir. Annotation yalnız "
        "`annotation_log.jsonl` dosyasına yazılır."
    )

    root = _root_from_env()
    refs = _load_json(root / "read_only_refs" / "artifact_pointers.json")
    session_dir = root / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ann_path = session_dir / "annotation_log.jsonl"
    assert_not_product_log(ann_path)
    gt_path = session_dir / "ground_truth_draft.json"
    meta_path = session_dir / "annotation_session.json"

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
            }
            _write_json(meta_path, sess)
            st.session_state.gt_session = sess
    if "selection" not in st.session_state:
        st.session_state.selection = {}
    if "proposal" not in st.session_state:
        st.session_state.proposal = None
    if "force_pause" not in st.session_state:
        st.session_state.force_pause = False

    sess = st.session_state.gt_session
    fps = float(sess["fps"])
    frame_count = int(sess["frame_count"])
    log = AnnotationLog(ann_path)

    base = empty_ground_truth(
        source_video_sha256=sess["source_video_sha256"],
        match_id=sess["match_id"],
        analysis_run_id=sess["analysis_run_id"],
        target_id=sess["target_id"],
        annotation_session_id=sess["annotation_session_id"],
        reviewer=sess["reviewer"],
        frame_count=frame_count,
        fps=fps,
    )
    gt = rebuild_gt_from_events(log.read_raw(), base_gt=base, fps=fps)
    _write_json(gt_path, gt)

    # Dense observations (read-only preprocess artifact)
    dens_full = load_dense_timeline(Path(refs["dense_bbox_timeline"]))
    dens = observations_for_component(dens_full)

    actives = active_intervals(gt)
    covered = 0
    for iv in actives:
        covered += int(iv["end_frame"]) - int(iv["start_frame"]) + 1
    remaining = max(0, frame_count - covered)
    current_state = "REVIEW REQUIRED"
    if st.session_state.proposal:
        current_state = "PROPOSAL"
    elif actives:
        # state at selection frame if any
        fi = st.session_state.selection.get("selected_frame_index")
        if fi is not None:
            for iv in actives:
                if int(iv["start_frame"]) <= int(fi) <= int(iv["end_frame"]):
                    current_state = iv["target_state"]
                    break

    c1, c2, c3 = st.columns(3)
    c1.metric("Current target state", current_state)
    c2.metric("Completed frames", f"{covered}/{frame_count}")
    c3.metric("Remaining frames", remaining)

    sel = st.session_state.selection or {}
    st.write(
        {
            "selected_raw_track_id": sel.get("selected_raw_track_id"),
            "selected_segment_id": sel.get("selected_segment_id"),
            "selected_frame_index": sel.get("selected_frame_index"),
            "proposal": (
                {
                    "start_frame": st.session_state.proposal.get("start_frame"),
                    "end_frame": st.session_state.proposal.get("end_frame"),
                    "raw_track_ids": st.session_state.proposal.get(
                        "associated_raw_track_ids"
                    ),
                }
                if st.session_state.proposal
                else None
            ),
        }
    )

    video = Path(refs["source_video"]).resolve()
    if not video.is_absolute() or not video.is_file():
        # relative to project
        project = Path(__file__).resolve().parents[4]
        video = (project / refs["source_video"]).resolve()
    proxy = ensure_interactive_review_proxy(
        source_video=video,
        source_video_sha256=refs["source_video_sha256"],
        output_path=session_dir / "interactive_review" / "source_proxy_960.mp4",
    )

    track_end = None
    if st.session_state.proposal:
        track_end = int(st.session_state.proposal["end_frame"])
    elif sel.get("selected_raw_track_id"):
        first, last, _, _ = track_span_from_observations(
            dens,
            raw_track_id=str(sel["selected_raw_track_id"]),
            segment_id=sel.get("selected_segment_id"),
        )
        track_end = last

    markers = [
        {
            "event_id": iv["annotation_id"],
            "label": f"{iv['target_state'][:12]}@{iv['start_frame']}",
            "frame_index": int(iv["start_frame"]),
        }
        for iv in actives[:40]
    ]

    click = interactive_video_review(
        video_data_url=proxy["data_url"],
        observations_by_frame=dens,
        markers=markers,
        selected_raw_track_id=sel.get("selected_raw_track_id"),
        selected_segment_id=sel.get("selected_segment_id"),
        track_end_frame=track_end,
        fps=fps,
        video_width=int(refs["resolution"]["width"]),
        video_height=int(refs["resolution"]["height"]),
        force_pause=bool(st.session_state.force_pause),
        seek_frame=(
            int(sel["selected_frame_index"])
            if sel.get("selected_frame_index") is not None and st.session_state.force_pause
            else None
        ),
        key="gt_interactive",
    )

    if isinstance(click, dict) and click.get("type") == "bbox_click":
        st.session_state.selection = {
            "selected_raw_track_id": click.get("raw_track_id"),
            "selected_segment_id": click.get("segment_id"),
            "selected_frame_index": click.get("frame_index"),
            "selected_bbox_xyxy": click.get("bbox_xyxy"),
            "candidate_id": click.get("candidate_id"),
        }
        first, last, bboxes, det_ids = track_span_from_observations(
            dens,
            raw_track_id=str(click.get("raw_track_id")),
            segment_id=click.get("segment_id"),
        )
        if first is not None and last is not None:
            st.session_state.proposal = proposal_from_track_span(
                raw_track_id=str(click.get("raw_track_id")),
                segment_id=click.get("segment_id"),
                start_frame=first,
                end_frame=last,
                fps=fps,
                detection_ids=det_ids,
                bbox_observations=bboxes,
            )
        st.session_state.force_pause = True
        st.success(
            f"Proposal: track {click.get('raw_track_id')} "
            f"[{first},{last}] — etiket seçin"
        )
        st.rerun()

    st.markdown("### Annotation actions")
    st.caption("Klavye kısayolları opsiyonel değildir zorunlu; butonlarla ilerleyin.")

    def _commit(state: str, *, wrong: bool = False) -> None:
        prop = st.session_state.proposal
        if not prop:
            st.error("Önce hedef bbox’a tıklayın (proposal gerekli)")
            return
        iv = build_annotation_interval(
            start_frame=int(prop["start_frame"]),
            end_frame=int(prop["end_frame"]),
            fps=fps,
            target_state=state,
            associated_detection_ids=list(prop.get("associated_detection_ids") or []),
            associated_raw_track_ids=list(prop.get("associated_raw_track_ids") or []),
            associated_segment_ids=list(prop.get("associated_segment_ids") or []),
            bbox_observations=list(prop.get("bbox_observations") or []),
            occlusion_state="occluded" if state == "TARGET_OCCLUDED" else None,
            visibility_confidence="high",
            reviewer_comment="ui_commit",
            active=True,
            provenance={
                "ui_profile": UI_PROFILE,
                "wrong_target_means_associated_is_wrong_track": wrong,
            },
        )
        # If overlapping an active interval, supersede by deactivating overlaps via new events
        for old in active_intervals(gt):
            if int(old["end_frame"]) < int(iv["start_frame"]) or int(old["start_frame"]) > int(
                iv["end_frame"]
            ):
                continue
            # supersede overlapping old
            deactivated = dict(old)
            deactivated["active"] = False
            log.append(
                build_annotation_event(
                    action="DEACTIVATE_INTERVAL",
                    interval=deactivated,
                    reviewer=sess["reviewer"],
                    annotation_session_id=sess["annotation_session_id"],
                    source_video_sha256=sess["source_video_sha256"],
                    match_id=sess["match_id"],
                    analysis_run_id=sess["analysis_run_id"],
                    target_id=sess["target_id"],
                    comment="superseded_by_overlap_replace",
                )
            )
            iv["supersedes_annotation_id"] = old["annotation_id"]
        log.append(
            build_annotation_event(
                action="APPEND_INTERVAL",
                interval=iv,
                reviewer=sess["reviewer"],
                annotation_session_id=sess["annotation_session_id"],
                source_video_sha256=sess["source_video_sha256"],
                match_id=sess["match_id"],
                analysis_run_id=sess["analysis_run_id"],
                target_id=sess["target_id"],
            )
        )
        st.session_state.proposal = None
        st.session_state.force_pause = False
        st.success(f"Yazıldı: {state} / {iv['annotation_id']}")
        st.rerun()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Correct target", type="primary"):
            _commit("TARGET_VISIBLE_ASSOCIATED")
        if st.button("Wrong target"):
            _commit("WRONG_TARGET_ASSIGNED", wrong=True)
    with b2:
        if st.button("Target visible but missed"):
            _commit("TARGET_VISIBLE_BUT_MISSED")
        if st.button("Occluded"):
            _commit("TARGET_OCCLUDED")
    with b3:
        if st.button("Out of frame"):
            _commit("TARGET_OUT_OF_FRAME")
        if st.button("Uncertain"):
            _commit("TARGET_UNCERTAIN")

    st.markdown("### Split / merge")
    split_frame = st.number_input(
        "Split frame",
        min_value=0,
        max_value=max(0, frame_count - 1),
        value=int(sel.get("selected_frame_index") or 0),
    )
    if st.button("Split interval at current frame"):
        # split active interval covering frame, or proposal
        target_iv = None
        for iv in actives:
            if int(iv["start_frame"]) < int(split_frame) <= int(iv["end_frame"]):
                target_iv = iv
                break
        if target_iv is None and st.session_state.proposal:
            target_iv = st.session_state.proposal
        if target_iv is None:
            st.error("Bölünecek interval yok")
        else:
            left, right = split_interval_at_frame(
                target_iv, split_frame=int(split_frame), fps=fps
            )
            # Uncommitted proposal: split locally only (no product/log contamination)
            if target_iv.get("active") is False or not any(
                a["annotation_id"] == target_iv.get("annotation_id") for a in actives
            ):
                st.session_state.proposal = right
                st.info(
                    f"Proposal split @ {split_frame}: "
                    f"left=[{left['start_frame']},{left['end_frame']}] "
                    f"right=[{right['start_frame']},{right['end_frame']}] "
                    "(sağ parça seçili; etiketleyin)"
                )
                st.session_state.force_pause = True
                st.rerun()
            left["supersedes_annotation_id"] = target_iv.get("annotation_id")
            right["supersedes_annotation_id"] = target_iv.get("annotation_id")
            ev = build_annotation_event(
                action="SPLIT_INTERVAL",
                interval=dict(target_iv),
                reviewer=sess["reviewer"],
                annotation_session_id=sess["annotation_session_id"],
                source_video_sha256=sess["source_video_sha256"],
                match_id=sess["match_id"],
                analysis_run_id=sess["analysis_run_id"],
                target_id=sess["target_id"],
            )
            ev["split"] = {"left": left, "right": right}
            log.append(ev)
            st.session_state.proposal = right
            st.session_state.force_pause = True
            st.success(f"Split @ {split_frame}")
            st.rerun()

    if st.button("Merge last two compatible active intervals"):
        if len(actives) < 2:
            st.error("En az iki active interval gerekli")
        else:
            a, b = actives[-2], actives[-1]
            try:
                merged = merge_intervals(a, b, fps=fps)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
            else:
                ev = build_annotation_event(
                    action="MERGE_INTERVALS",
                    interval=merged,
                    reviewer=sess["reviewer"],
                    annotation_session_id=sess["annotation_session_id"],
                    source_video_sha256=sess["source_video_sha256"],
                    match_id=sess["match_id"],
                    analysis_run_id=sess["analysis_run_id"],
                    target_id=sess["target_id"],
                )
                ev["merge"] = {"deactivate_ids": [a["annotation_id"], b["annotation_id"]]}
                log.append(ev)
                st.success("Merged")
                st.rerun()

    st.markdown("### Annotation timeline")
    st.write(
        [
            {
                "annotation_id": iv["annotation_id"],
                "start_frame": iv["start_frame"],
                "end_frame": iv["end_frame"],
                "target_state": iv["target_state"],
                "raw_track_ids": iv.get("associated_raw_track_ids"),
                "active": iv.get("active"),
            }
            for iv in (gt.get("intervals") or [])[-30:]
        ]
    )

    with st.expander("Advanced / validation"):
        st.caption("Product decision history / Timeline Approval / gallery gizli.")
        report = validate_ground_truth(
            gt,
            expected_source_sha256=refs["source_video_sha256"],
            frame_count=frame_count,
            fps=fps,
        )
        st.write(report)
        if st.button("Validate & freeze draft for overlay"):
            _write_json(gt_path, gt)
            _write_json(session_dir / "validation_report.json", report)
            if report["ok"]:
                st.success("Validation OK — acceptance için overlay üretilebilir")
            else:
                st.error("Validation failed — boşluklar otomatik etiketlenmez")

        if st.button("Accept ground truth (after overlay review)"):
            report = validate_ground_truth(
                gt,
                expected_source_sha256=refs["source_video_sha256"],
                frame_count=frame_count,
                fps=fps,
            )
            if not report["ok"]:
                st.error("Acceptance blocked: " + "; ".join(report["errors"][:5]))
            else:
                ev = build_annotation_event(
                    action="ACCEPT_GROUND_TRUTH",
                    interval={"annotation_id": "accept", "start_frame": 0, "end_frame": 0, "target_state": "TARGET_UNCERTAIN", "active": False},
                    reviewer=sess["reviewer"],
                    annotation_session_id=sess["annotation_session_id"],
                    source_video_sha256=sess["source_video_sha256"],
                    match_id=sess["match_id"],
                    analysis_run_id=sess["analysis_run_id"],
                    target_id=sess["target_id"],
                    comment="human_acceptance",
                )
                # fix interval to satisfy schema minimally
                ev["interval"] = build_annotation_interval(
                    start_frame=0,
                    end_frame=0,
                    fps=fps,
                    target_state="TARGET_UNCERTAIN",
                    active=False,
                    provenance={"acceptance_marker": True},
                )
                log.append(ev)
                accepted = rebuild_gt_from_events(log.read_raw(), base_gt=base, fps=fps)
                accepted["accepted"] = True
                _write_json(session_dir / "accepted_ground_truth.json", accepted)
                st.success("Accepted GT yazıldı (metrics script ayrı çalıştırılmalı)")
                st.rerun()


if __name__ == "__main__":
    run_app()
