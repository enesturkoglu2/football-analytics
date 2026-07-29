"""TARGET_FAILURE_WINDOW_PILOT schema and candidate failure-window generation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from football_analytics.reid.golden_clip.window_index import DEFAULT_WINDOW_RADIUS_FRAMES

PILOT_SCHEMA = "target_failure_window_pilot_v1"
COVERAGE_SCOPE = "FAILURE_WINDOW_PILOT"

PILOT_LABELS = frozenset(
    {
        "RAW_TRACK_FRAGMENT_SAME_TARGET",
        "SHORT_OCCLUSION_FRAGMENTATION",
        "WRONG_PLAYER_ID_SWITCH",
        "TARGET_VISIBLE_BUT_DETECTION_MISSED",
        "OUT_OF_FRAME",
        "BORDER_EXIT_REENTRY",
        "UNCERTAIN",
    }
)

# Candidate continuation selection required for these labels (R1.2)
CANDIDATE_REQUIRED_LABELS = frozenset(
    {
        "RAW_TRACK_FRAGMENT_SAME_TARGET",
        "SHORT_OCCLUSION_FRAGMENTATION",
        "BORDER_EXIT_REENTRY",
    }
)

NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT = "NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_pilot_doc(
    *,
    source_video_sha256: str,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    annotation_session_id: str,
    reviewer: str,
) -> dict[str, Any]:
    return {
        "schema_version": PILOT_SCHEMA,
        "coverage_scope": COVERAGE_SCOPE,
        "source_video_sha256": source_video_sha256,
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "annotation_session_id": annotation_session_id,
        "reviewer": reviewer,
        "created_at": _utc(),
        "accepted_full_gt": False,
        "windows": [],
        "note_tr": (
            "Pilot failure-window annotation; tam 45.29s accepted GT değildir. "
            "IDF1/full recall hesaplanmaz."
        ),
    }


def build_pilot_label_event(
    *,
    window: Mapping[str, Any],
    label: str,
    reviewer: str,
    annotation_session_id: str,
    source_video_sha256: str,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    selected_next_raw_track_id: str | None = None,
    comment: str = "",
    event_uuid: str | None = None,
    failure_window_id: str | None = None,
    ui_mode: str = "static",
) -> dict[str, Any]:
    if label not in PILOT_LABELS:
        raise ValueError(f"invalid pilot label: {label}")
    if label in CANDIDATE_REQUIRED_LABELS and not selected_next_raw_track_id:
        raise ValueError(f"label {label} requires selected continuation candidate")
    wid = failure_window_id or window.get("window_id") or window.get("event_id")
    start_frame = int(window.get("start_frame") or window.get("track_end_frame") or 0)
    end_frame = int(window.get("end_frame") or start_frame)
    return {
        "schema_version": "target_gt_annotation_event_v1",
        "event_id": f"gtevt_{uuid.uuid4().hex[:12]}",
        "event_uuid": event_uuid or f"euuid_{uuid.uuid4().hex[:16]}",
        "action": "PILOT_FAILURE_WINDOW_LABEL",
        "created_at": _utc(),
        "reviewer": reviewer,
        "annotation_session_id": annotation_session_id,
        "source_video_sha256": source_video_sha256,
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "comment": comment,
        "coverage_scope": COVERAGE_SCOPE,
        "ui_mode": ui_mode,
        "failure_window_id": wid,
        "interval": {
            "annotation_id": f"ann_{uuid.uuid4().hex[:12]}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": float(window.get("start_time") or window.get("track_end_time") or 0.0),
            "end_time": float(window.get("end_time") or window.get("track_end_time") or 0.0),
            "target_state": "TARGET_UNCERTAIN",
            "associated_detection_ids": [],
            "associated_raw_track_ids": [
                str(window.get("previous_raw_track_id") or "")
            ]
            + ([str(selected_next_raw_track_id)] if selected_next_raw_track_id else []),
            "associated_segment_ids": [],
            "bbox_observations": [],
            "true_target_bbox_observations": [],
            "occlusion_state": None,
            "visibility_confidence": "pilot",
            "reviewer_comment": comment,
            "supersedes_annotation_id": None,
            "active": True,
            "provenance": {
                "coverage_scope": COVERAGE_SCOPE,
                "pilot_label": label,
                "window_id": wid,
                "selected_next_raw_track_id": selected_next_raw_track_id,
                "incomplete_ui_freeze_event": False,
                "ui_mode": ui_mode,
            },
        },
        "pilot": {
            "label": label,
            "window": dict(window),
            "selected_next_raw_track_id": selected_next_raw_track_id,
            "failure_window_id": wid,
        },
    }


def generate_failure_windows(
    *,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_raw_track_id: str,
    selected_segment_id: str | None,
    track_first: int,
    track_last: int,
    fps: float,
    frame_count: int,
    max_windows: int = 8,
    radius: int = DEFAULT_WINDOW_RADIUS_FRAMES,
) -> list[dict[str, Any]]:
    """Heuristic candidate windows around track end / nearby births (no auto identity)."""
    windows: list[dict[str, Any]] = []
    lost = int(track_last)
    # 1) selected track end
    windows.append(
        _mk_window(
            kind="selected_raw_track_end",
            center=lost,
            fps=fps,
            frame_count=frame_count,
            radius=radius,
            previous_raw_track_id=selected_raw_track_id,
            previous_segment_id=selected_segment_id,
            observations_by_frame=observations_by_frame,
        )
    )
    # 2) nearby new track births after loss
    births: dict[str, int] = {}
    for fi in range(lost + 1, min(frame_count, lost + 90)):
        for r in observations_by_frame.get(str(fi), []) or []:
            tid = str(r.get("raw_track_id"))
            if tid == str(selected_raw_track_id):
                continue
            if tid not in births:
                births[tid] = fi
    # rank births by spatial proximity to last selected bbox
    last_bbox = None
    for r in observations_by_frame.get(str(lost), []) or []:
        if str(r.get("raw_track_id")) == str(selected_raw_track_id):
            last_bbox = list(r["bbox_xyxy"])
            break
    ranked = []
    for tid, birth_fi in births.items():
        rows = observations_by_frame.get(str(birth_fi), []) or []
        entry = next((r for r in rows if str(r.get("raw_track_id")) == tid), None)
        dist = 1e9
        if entry and last_bbox:
            cx = (float(entry["bbox_xyxy"][0]) + float(entry["bbox_xyxy"][2])) / 2
            cy = (float(entry["bbox_xyxy"][1]) + float(entry["bbox_xyxy"][3])) / 2
            px = (float(last_bbox[0]) + float(last_bbox[2])) / 2
            py = (float(last_bbox[1]) + float(last_bbox[3])) / 2
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        # detection continuity: any det in gap?
        gap_dets = sum(
            1
            for g in range(lost + 1, birth_fi)
            if observations_by_frame.get(str(g))
        )
        ranked.append((dist, birth_fi, tid, gap_dets, entry))
    ranked.sort(key=lambda x: (x[0], x[1]))
    for dist, birth_fi, tid, gap_dets, entry in ranked[: max_windows - 1]:
        windows.append(
            _mk_window(
                kind="nearby_new_track_birth",
                center=birth_fi,
                fps=fps,
                frame_count=frame_count,
                radius=radius,
                previous_raw_track_id=selected_raw_track_id,
                previous_segment_id=selected_segment_id,
                candidate_raw_track_ids=[tid],
                time_gap_frames=birth_fi - lost,
                detection_continuity_frames=gap_dets,
                spatial_displacement_px=dist if dist < 1e8 else None,
                observations_by_frame=observations_by_frame,
                bbox_overlap=_overlap_score(
                    observations_by_frame, lost, birth_fi, selected_raw_track_id, tid
                ),
            )
        )
    # 3) mid-track absence pockets (selected track missing while others present)
    absent_start = None
    for fi in range(int(track_first), int(track_last) + 1):
        rows = observations_by_frame.get(str(fi), []) or []
        present = any(str(r.get("raw_track_id")) == str(selected_raw_track_id) for r in rows)
        others = len(rows) > 0
        if (not present) and others:
            if absent_start is None:
                absent_start = fi
        else:
            if absent_start is not None and fi - absent_start >= 3:
                windows.append(
                    _mk_window(
                        kind="selected_track_absence",
                        center=(absent_start + fi - 1) // 2,
                        fps=fps,
                        frame_count=frame_count,
                        radius=min(radius, 30),
                        previous_raw_track_id=selected_raw_track_id,
                        previous_segment_id=selected_segment_id,
                        observations_by_frame=observations_by_frame,
                    )
                )
            absent_start = None
        if len(windows) >= max_windows:
            break
    # dedupe by center proximity
    deduped: list[dict[str, Any]] = []
    for w in windows:
        if any(abs(int(w["center_frame"]) - int(d["center_frame"])) < 10 for d in deduped):
            continue
        deduped.append(w)
        if len(deduped) >= max_windows:
            break
    return deduped


def _overlap_score(
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    fi_a: int,
    fi_b: int,
    tid_a: str,
    tid_b: str,
) -> float | None:
    def bbox(fi: int, tid: str):
        for r in observations_by_frame.get(str(fi), []) or []:
            if str(r.get("raw_track_id")) == str(tid):
                return list(r["bbox_xyxy"])
        return None

    a = bbox(fi_a, tid_a)
    b = bbox(fi_b, tid_b)
    if not a or not b:
        return None
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _mk_window(
    *,
    kind: str,
    center: int,
    fps: float,
    frame_count: int,
    radius: int,
    previous_raw_track_id: str,
    previous_segment_id: str | None,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_raw_track_ids: Sequence[str] | None = None,
    time_gap_frames: int | None = None,
    detection_continuity_frames: int | None = None,
    spatial_displacement_px: float | None = None,
    bbox_overlap: float | None = None,
) -> dict[str, Any]:
    c = max(0, min(int(center), int(frame_count) - 1))
    start = max(0, c - radius)
    end = min(int(frame_count) - 1, c + radius)
    # possible next tracks active in window after previous end-ish
    next_ids: list[str] = []
    for fi in range(start, end + 1):
        for r in observations_by_frame.get(str(fi), []) or []:
            tid = str(r.get("raw_track_id"))
            if tid != str(previous_raw_track_id) and tid not in next_ids:
                next_ids.append(tid)
            if len(next_ids) >= 12:
                break
        if len(next_ids) >= 12:
            break
    if candidate_raw_track_ids:
        for tid in candidate_raw_track_ids:
            if tid not in next_ids:
                next_ids.insert(0, str(tid))
    return {
        "window_id": f"fw_{uuid.uuid4().hex[:10]}",
        "kind": kind,
        "center_frame": c,
        "start_frame": start,
        "end_frame": end,
        "start_time": start / fps if fps else 0.0,
        "end_time": (end + 1) / fps if fps else 0.0,
        "previous_raw_track_id": str(previous_raw_track_id),
        "previous_segment_id": previous_segment_id,
        "possible_next_raw_tracks": next_ids,
        "time_gap_frames": time_gap_frames,
        "detection_continuity_frames": detection_continuity_frames,
        "bbox_overlap": bbox_overlap,
        "image_position_displacement_px": spatial_displacement_px,
        "tracker_variant": "A_current_bytetrack",
        "auto_identity_forbidden": True,
    }


def summarize_pilot_labels(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {k: 0 for k in PILOT_LABELS}
    reps: dict[str, list[dict[str, Any]]] = {k: [] for k in PILOT_LABELS}
    for ev in events:
        if ev.get("action") != "PILOT_FAILURE_WINDOW_LABEL":
            continue
        if ev.get("coverage_scope") != COVERAGE_SCOPE:
            continue
        label = str((ev.get("pilot") or {}).get("label") or "")
        if label not in counts:
            continue
        counts[label] += 1
        w = (ev.get("pilot") or {}).get("window") or {}
        if len(reps[label]) < 5:
            reps[label].append(
                {
                    "window_id": w.get("window_id"),
                    "center_frame": w.get("center_frame"),
                    "start_time": w.get("start_time"),
                    "kind": w.get("kind"),
                }
            )
    # map to taxonomy names used by R0/R1 / R1.2
    taxonomy = {
        "RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION": {
            "count": counts["RAW_TRACK_FRAGMENT_SAME_TARGET"],
            "representative_timestamps": reps["RAW_TRACK_FRAGMENT_SAME_TARGET"],
        },
        "SHORT_OCCLUSION_FRAGMENTATION": {
            "count": counts["SHORT_OCCLUSION_FRAGMENTATION"],
            "representative_timestamps": reps["SHORT_OCCLUSION_FRAGMENTATION"],
        },
        "WRONG_PLAYER_ID_SWITCH": {
            "count": counts["WRONG_PLAYER_ID_SWITCH"],
            "representative_timestamps": reps["WRONG_PLAYER_ID_SWITCH"],
        },
        "DETECTION_MISS": {
            "count": counts["TARGET_VISIBLE_BUT_DETECTION_MISSED"],
            "representative_timestamps": reps["TARGET_VISIBLE_BUT_DETECTION_MISSED"],
        },
        "OUT_OF_FRAME": {
            "count": counts["OUT_OF_FRAME"],
            "representative_timestamps": reps["OUT_OF_FRAME"],
        },
        "BORDER_EXIT_REENTRY": {
            "count": counts["BORDER_EXIT_REENTRY"],
            "representative_timestamps": reps["BORDER_EXIT_REENTRY"],
        },
        "UNCERTAIN": {
            "count": counts["UNCERTAIN"],
            "representative_timestamps": reps["UNCERTAIN"],
        },
    }
    labeled = sum(counts.values())
    continuation_ids = []
    for ev in events:
        if ev.get("action") != "PILOT_FAILURE_WINDOW_LABEL":
            continue
        tid = (ev.get("pilot") or {}).get("selected_next_raw_track_id")
        if tid and tid not in continuation_ids:
            continuation_ids.append(str(tid))
    return {
        "coverage_scope": COVERAGE_SCOPE,
        "labeled_window_count": labeled,
        "label_counts": counts,
        "taxonomy": taxonomy,
        "selected_continuation_track_ids": continuation_ids,
        "available_old_identity_signals": [
            "quality",
            "kit",
            "purity",
            "manual_segments",
            "linking_policy_manual_only",
        ],
        "full_metrics": {
            "target_idf1": NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
            "full_video_recall": NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
            "target_visible_coverage": NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
            "false_target_identity_switch_count": NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
        },
    }


def choose_pilot_next_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    labeled = int(summary.get("labeled_window_count") or 0)
    if labeled < 3:
        return {
            "exact_next_gate": "TARGET_FAILURE_WINDOW_PILOT_MORE_USER_EVIDENCE_REQUIRED",
            "rationale_tr": "En az 3 insan etiketli failure window gerekli.",
            "labeled_window_count": labeled,
        }
    tax = summary.get("taxonomy") or {}
    frag = int((tax.get("RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION") or {}).get("count") or 0)
    occ = int((tax.get("SHORT_OCCLUSION_FRAGMENTATION") or {}).get("count") or 0)
    miss = int((tax.get("DETECTION_MISS") or {}).get("count") or 0)
    wrong = int((tax.get("WRONG_PLAYER_ID_SWITCH") or {}).get("count") or 0)
    border = int((tax.get("BORDER_EXIT_REENTRY") or {}).get("count") or 0)
    # Majority / dominant class (R1.2 exact gate selection)
    scores = {
        "EXISTING_TRACKLET_STITCHING_INTEGRATION": frag,
        "TARGET_CONDITIONED_SHORT_OCCLUSION_BRIDGE": occ,
        "TARGET_DETECTION_RECALL_REMEDIATION": miss,
        "TARGET_ID_STATE_AND_OBSERVATION_LAYER": wrong + border,
    }
    best_gate = max(scores, key=lambda k: scores[k])
    if scores[best_gate] <= 0:
        gate = "TARGET_ID_STATE_AND_OBSERVATION_LAYER"
        rationale = "Pilot etiketler baskın sınıf göstermiyor; state katmanı güvenli kapı."
    elif best_gate == "EXISTING_TRACKLET_STITCHING_INTEGRATION":
        gate = best_gate
        rationale = "RAW_TRACK_FRAGMENT_SAME_TARGET baskın."
    elif best_gate == "TARGET_CONDITIONED_SHORT_OCCLUSION_BRIDGE":
        gate = best_gate
        rationale = "SHORT_OCCLUSION_FRAGMENTATION baskın."
    elif best_gate == "TARGET_DETECTION_RECALL_REMEDIATION":
        gate = best_gate
        rationale = "TARGET_VISIBLE_BUT_DETECTION_MISSED baskın."
    else:
        gate = "TARGET_ID_STATE_AND_OBSERVATION_LAYER"
        rationale = "Karışık / wrong-switch / border; persistent target state katmanı."
    return {
        "exact_next_gate": gate,
        "rationale_tr": rationale,
        "labeled_window_count": labeled,
        "single_gate_only": True,
        "class_scores": scores,
    }
