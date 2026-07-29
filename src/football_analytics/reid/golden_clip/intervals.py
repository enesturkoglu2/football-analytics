"""Interval split/merge and GT rebuild from append-only annotation log."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.golden_clip.schema import (
    GoldenClipError,
    build_annotation_interval,
    new_annotation_id,
    validate_annotation_interval,
)


def proposal_from_track_span(
    *,
    raw_track_id: str,
    segment_id: str | None,
    start_frame: int,
    end_frame: int,
    fps: float,
    detection_ids: Sequence[str] | None = None,
    bbox_observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """UI proposal for the selected raw track's continuous span (not yet committed)."""
    return build_annotation_interval(
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        target_state="TARGET_UNCERTAIN",
        associated_detection_ids=list(detection_ids or []),
        associated_raw_track_ids=[str(raw_track_id)],
        associated_segment_ids=[str(segment_id)] if segment_id else [],
        bbox_observations=list(bbox_observations or []),
        visibility_confidence="proposal",
        active=False,
        provenance={"kind": "track_assisted_proposal", "committed": False},
    )


def split_interval_at_frame(
    interval: Mapping[str, Any],
    *,
    split_frame: int,
    fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split [start,end] into [start, split-1] and [split, end]."""
    start = int(interval["start_frame"])
    end = int(interval["end_frame"])
    sf = int(split_frame)
    if sf <= start or sf > end:
        raise GoldenClipError(
            f"split_frame {sf} must be within ({start}, {end}]"
        )
    left_obs = [
        o
        for o in (interval.get("bbox_observations") or [])
        if int(o.get("frame_index", -1)) < sf
    ]
    right_obs = [
        o
        for o in (interval.get("bbox_observations") or [])
        if int(o.get("frame_index", -1)) >= sf
    ]
    left = build_annotation_interval(
        annotation_id=new_annotation_id(),
        start_frame=start,
        end_frame=sf - 1,
        fps=fps,
        target_state=str(interval["target_state"]),
        associated_detection_ids=list(interval.get("associated_detection_ids") or []),
        associated_raw_track_ids=list(interval.get("associated_raw_track_ids") or []),
        associated_segment_ids=list(interval.get("associated_segment_ids") or []),
        bbox_observations=left_obs,
        true_target_bbox_observations=list(
            interval.get("true_target_bbox_observations") or []
        ),
        occlusion_state=interval.get("occlusion_state"),
        visibility_confidence=str(interval.get("visibility_confidence") or "medium"),
        reviewer_comment=str(interval.get("reviewer_comment") or ""),
        supersedes_annotation_id=str(interval["annotation_id"]),
        active=True,
        provenance={
            **dict(interval.get("provenance") or {}),
            "split_from": interval["annotation_id"],
            "split_role": "left",
        },
    )
    right = build_annotation_interval(
        annotation_id=new_annotation_id(),
        start_frame=sf,
        end_frame=end,
        fps=fps,
        target_state=str(interval["target_state"]),
        associated_detection_ids=list(interval.get("associated_detection_ids") or []),
        associated_raw_track_ids=list(interval.get("associated_raw_track_ids") or []),
        associated_segment_ids=list(interval.get("associated_segment_ids") or []),
        bbox_observations=right_obs,
        true_target_bbox_observations=list(
            interval.get("true_target_bbox_observations") or []
        ),
        occlusion_state=interval.get("occlusion_state"),
        visibility_confidence=str(interval.get("visibility_confidence") or "medium"),
        reviewer_comment=str(interval.get("reviewer_comment") or ""),
        supersedes_annotation_id=str(interval["annotation_id"]),
        active=True,
        provenance={
            **dict(interval.get("provenance") or {}),
            "split_from": interval["annotation_id"],
            "split_role": "right",
        },
    )
    return left, right


def can_merge_intervals(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Adjacent same-state intervals with same association may merge."""
    if str(a.get("target_state")) != str(b.get("target_state")):
        return False
    if list(a.get("associated_raw_track_ids") or []) != list(
        b.get("associated_raw_track_ids") or []
    ):
        return False
    if list(a.get("associated_segment_ids") or []) != list(
        b.get("associated_segment_ids") or []
    ):
        return False
    # adjacent (inclusive frames)
    return int(a["end_frame"]) + 1 == int(b["start_frame"]) or int(
        b["end_frame"]
    ) + 1 == int(a["start_frame"])


def merge_intervals(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    fps: float,
) -> dict[str, Any]:
    if not can_merge_intervals(a, b):
        raise GoldenClipError("intervals are not mergeable")
    first, second = (a, b) if int(a["start_frame"]) <= int(b["start_frame"]) else (b, a)
    obs = list(first.get("bbox_observations") or []) + list(
        second.get("bbox_observations") or []
    )
    return build_annotation_interval(
        annotation_id=new_annotation_id(),
        start_frame=int(first["start_frame"]),
        end_frame=int(second["end_frame"]),
        fps=fps,
        target_state=str(first["target_state"]),
        associated_detection_ids=sorted(
            set(list(first.get("associated_detection_ids") or [])
                + list(second.get("associated_detection_ids") or []))
        ),
        associated_raw_track_ids=list(first.get("associated_raw_track_ids") or []),
        associated_segment_ids=list(first.get("associated_segment_ids") or []),
        bbox_observations=obs,
        true_target_bbox_observations=list(
            first.get("true_target_bbox_observations") or []
        )
        + list(second.get("true_target_bbox_observations") or []),
        occlusion_state=first.get("occlusion_state"),
        visibility_confidence=str(first.get("visibility_confidence") or "medium"),
        reviewer_comment="merged",
        supersedes_annotation_id=None,
        active=True,
        provenance={
            "merged_from": [first["annotation_id"], second["annotation_id"]],
        },
    )


def rebuild_gt_from_events(
    events: Sequence[Mapping[str, Any]],
    *,
    base_gt: Mapping[str, Any],
    fps: float,
) -> dict[str, Any]:
    """Apply append-only events to produce current GT document.

    Supported actions:
    - APPEND_INTERVAL: add active interval (may supersede prior by id)
    - SUPERSEDE_INTERVAL: deactivate old, add new
    - SPLIT_INTERVAL: deactivate old, add left+right
    - MERGE_INTERVALS: deactivate two, add merged
    - DEACTIVATE_INTERVAL: set active=False
    """
    gt = dict(base_gt)
    by_id: dict[str, dict[str, Any]] = {
        str(iv["annotation_id"]): dict(iv) for iv in (gt.get("intervals") or [])
    }
    revision = int(gt.get("revision") or 0)

    for ev in events:
        action = str(ev.get("action") or "")
        if action == "APPEND_INTERVAL":
            iv = validate_annotation_interval(ev["interval"], fps=fps)
            sid = iv.get("supersedes_annotation_id")
            if sid and sid in by_id:
                by_id[sid]["active"] = False
            by_id[iv["annotation_id"]] = iv
            revision += 1
        elif action == "SUPERSEDE_INTERVAL":
            iv = validate_annotation_interval(ev["interval"], fps=fps)
            old = iv.get("supersedes_annotation_id")
            if not old:
                raise GoldenClipError("SUPERSEDE_INTERVAL requires supersedes_annotation_id")
            if old in by_id:
                by_id[old]["active"] = False
            by_id[iv["annotation_id"]] = iv
            revision += 1
        elif action == "SPLIT_INTERVAL":
            # event.interval is the original; metadata has left/right
            meta = ev.get("split") or {}
            old_id = str((ev.get("interval") or {}).get("annotation_id") or "")
            if old_id in by_id:
                by_id[old_id]["active"] = False
            left = validate_annotation_interval(meta["left"], fps=fps)
            right = validate_annotation_interval(meta["right"], fps=fps)
            by_id[left["annotation_id"]] = left
            by_id[right["annotation_id"]] = right
            revision += 1
        elif action == "MERGE_INTERVALS":
            meta = ev.get("merge") or {}
            for oid in meta.get("deactivate_ids") or []:
                if oid in by_id:
                    by_id[oid]["active"] = False
            merged = validate_annotation_interval(ev["interval"], fps=fps)
            by_id[merged["annotation_id"]] = merged
            revision += 1
        elif action == "DEACTIVATE_INTERVAL":
            oid = str((ev.get("interval") or {}).get("annotation_id") or "")
            if oid in by_id:
                by_id[oid]["active"] = False
            revision += 1
        elif action == "ACCEPT_GROUND_TRUTH":
            gt["accepted"] = True
            gt["accepted_at"] = ev.get("created_at")
            gt["accepted_by"] = ev.get("reviewer")
            revision += 1
        elif action in {
            "PILOT_FAILURE_WINDOW_LABEL",
            "INCOMPLETE_UI_FREEZE_EVENT",
            "QUALIFY_INCOMPLETE",
        }:
            # Pilot / incomplete events are stored append-only but do not mutate
            # full-coverage ground-truth intervals.
            continue
        else:
            raise GoldenClipError(f"unknown annotation action: {action!r}")

    intervals = sorted(by_id.values(), key=lambda r: (int(r["start_frame"]), r["annotation_id"]))
    gt["intervals"] = intervals
    gt["revision"] = revision
    return gt


def track_span_from_observations(
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    raw_track_id: str,
    segment_id: str | None = None,
) -> tuple[int | None, int | None, list[dict[str, Any]], list[str]]:
    """Compute contiguous min/max frames for a raw track from dense obs."""
    first = last = None
    bboxes: list[dict[str, Any]] = []
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
            bboxes.append(
                {
                    "frame_index": fi,
                    "bbox_xyxy": list(r["bbox_xyxy"]),
                    "detection_id": r.get("detection_id"),
                    "raw_track_id": str(r.get("raw_track_id")),
                    "segment_id": r.get("segment_id"),
                }
            )
            if r.get("detection_id"):
                det_ids.append(str(r["detection_id"]))
    return first, last, bboxes, det_ids
