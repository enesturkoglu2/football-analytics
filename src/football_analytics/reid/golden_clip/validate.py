"""Validate golden-clip ground truth coverage and consistency."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.golden_clip import SCHEMA_GT
from football_analytics.reid.golden_clip.schema import GoldenClipError, active_intervals, validate_target_state


def validate_ground_truth(
    gt: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    frame_count: int,
    fps: float,
    detection_ids_by_frame: Mapping[int, set[str]] | None = None,
) -> dict[str, Any]:
    """Fail-closed validation before acceptance.

    Does NOT auto-fill gaps. TARGET_UNCERTAIN counts as coverage.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if gt.get("schema_version") != SCHEMA_GT:
        errors.append(f"schema_version must be {SCHEMA_GT}")
    if str(gt.get("source_video_sha256")) != str(expected_source_sha256):
        errors.append("source_video_sha256 mismatch")
    if int(gt.get("frame_count") or -1) != int(frame_count):
        errors.append("frame_count mismatch")

    actives = active_intervals(gt)
    # supersession chain: every supersedes_annotation_id must exist
    all_ids = {str(iv["annotation_id"]) for iv in (gt.get("intervals") or [])}
    for iv in gt.get("intervals") or []:
        sid = iv.get("supersedes_annotation_id")
        if sid and str(sid) not in all_ids:
            errors.append(f"broken supersession: {iv['annotation_id']} -> {sid}")
        try:
            validate_target_state(iv.get("target_state"))
        except GoldenClipError as exc:
            errors.append(str(exc))
        start = int(iv["start_frame"])
        end = int(iv["end_frame"])
        if start < 0 or end >= frame_count or end < start:
            errors.append(
                f"invalid frame range for {iv['annotation_id']}: [{start},{end}]"
            )

    # overlap among active
    sorted_act = sorted(actives, key=lambda r: (int(r["start_frame"]), int(r["end_frame"])))
    for i, a in enumerate(sorted_act):
        for b in sorted_act[i + 1 :]:
            if int(b["start_frame"]) > int(a["end_frame"]):
                break
            if int(a["start_frame"]) <= int(b["end_frame"]) and int(b["start_frame"]) <= int(
                a["end_frame"]
            ):
                errors.append(
                    f"overlapping active annotations: {a['annotation_id']} vs {b['annotation_id']}"
                )

    # coverage: every frame in [0, frame_count-1] must be in exactly one active interval
    covered = [False] * frame_count
    for iv in sorted_act:
        for fi in range(int(iv["start_frame"]), int(iv["end_frame"]) + 1):
            if 0 <= fi < frame_count:
                covered[fi] = True
    gaps = [i for i, ok in enumerate(covered) if not ok]
    if gaps:
        # compress gap ranges for report
        ranges = _compress_frames(gaps)
        errors.append(f"coverage gaps (not auto-labeled): {ranges[:20]}")

    # associated detections exist on frames when provided
    if detection_ids_by_frame is not None:
        for iv in sorted_act:
            for det_id in iv.get("associated_detection_ids") or []:
                found = False
                for fi in range(int(iv["start_frame"]), int(iv["end_frame"]) + 1):
                    if str(det_id) in detection_ids_by_frame.get(fi, set()):
                        found = True
                        break
                if not found and iv.get("target_state") in {
                    "TARGET_VISIBLE_ASSOCIATED",
                    "WRONG_TARGET_ASSIGNED",
                }:
                    warnings.append(
                        f"associated detection {det_id} not found in interval frames "
                        f"for {iv['annotation_id']}"
                    )

    uncertain_frames = 0
    for iv in sorted_act:
        if iv.get("target_state") == "TARGET_UNCERTAIN":
            uncertain_frames += int(iv["end_frame"]) - int(iv["start_frame"]) + 1

    ok = len(errors) == 0
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "active_interval_count": len(sorted_act),
        "covered_frames": sum(1 for x in covered if x),
        "frame_count": frame_count,
        "full_coverage": len(gaps) == 0 and ok,
        "uncertain_frame_count": uncertain_frames,
        "uncertain_counts_as_coverage": True,
        "uncertain_excluded_from_correctness_denominator_note": (
            "TARGET_UNCERTAIN is coverage-valid but excluded from precision/recall "
            "correctness denominator when evaluating tracker assignment."
        ),
    }


def _compress_frames(frames: Sequence[int]) -> list[list[int]]:
    if not frames:
        return []
    out: list[list[int]] = []
    start = prev = int(frames[0])
    for f in list(frames)[1:]:
        f = int(f)
        if f == prev + 1:
            prev = f
            continue
        out.append([start, prev])
        start = prev = f
    out.append([start, prev])
    return out
