"""Selection visualization helpers (Streamlit-independent)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SPARSE_PACKAGE_NOTICE = (
    "This review package contains sparse tracklet observations; it is not a "
    "continuous tracking preview."
)
NOT_VISIBLE_MESSAGE = "Selected tracklet is not visible in this frame."
TRACKING_NOT_RUN_NOTICE = (
    "Bu ekran videoyu yeniden takip etmez. Seçiminiz, önceden üretilmiş bir "
    "tracklet'i hedef oyuncu olarak insan kararıyla kaydeder."
)
SUBMIT_SUCCESS_MESSAGE = "Decision recorded. Tracking was not run."
SELECTED_LABEL = "SELECTED TARGET CANDIDATE"
RANK_UNAVAILABLE_LABEL = "Appearance ranking unavailable"


def appearance_rank_label(candidate: Mapping[str, Any]) -> str:
    rank = candidate.get("appearance_rank")
    if rank is None:
        return RANK_UNAVAILABLE_LABEL
    return f"Appearance rank {int(rank)}"


def observation_frames(candidate: Mapping[str, Any]) -> list[int]:
    frames = sorted(
        {
            int(ref["frame_index"])
            for ref in candidate.get("bbox_references") or []
        }
    )
    return frames


def is_sparse_observations(
    candidates: Sequence[Mapping[str, Any]],
    *,
    review_window_start: int,
    review_window_end: int,
) -> bool:
    window = max(0, int(review_window_end) - int(review_window_start) + 1)
    if window <= 1:
        return False
    for cand in candidates:
        frames = observation_frames(cand)
        if len(frames) < window:
            return True
    return False


def bbox_for_segment_on_frame(
    *,
    segment_id: str,
    frame_index: int,
    candidates: Sequence[Mapping[str, Any]],
    observation_lookup: Mapping[tuple[str, int], Sequence[float]] | None = None,
) -> list[float] | None:
    """Return bbox for segment at frame from manifest refs or optional obs index."""
    for cand in candidates:
        if cand.get("segment_id") != segment_id:
            continue
        for ref in cand.get("bbox_references") or []:
            if int(ref["frame_index"]) == int(frame_index):
                return [float(v) for v in ref["bbox_xyxy"]]
    if observation_lookup is not None:
        hit = observation_lookup.get((segment_id, int(frame_index)))
        if hit is not None:
            return [float(v) for v in hit]
    return None


def selection_visibility(
    *,
    selection: Mapping[str, Any] | None,
    frame_index: int,
    candidates: Sequence[Mapping[str, Any]],
    observation_lookup: Mapping[tuple[str, int], Sequence[float]] | None = None,
) -> dict[str, Any]:
    if not selection:
        return {
            "has_selection": False,
            "visible": False,
            "bbox_xyxy": None,
            "message": None,
            "label": None,
        }
    segment_id = str(selection.get("selected_segment_id") or "")
    bbox = bbox_for_segment_on_frame(
        segment_id=segment_id,
        frame_index=frame_index,
        candidates=candidates,
        observation_lookup=observation_lookup,
    )
    if bbox is None:
        return {
            "has_selection": True,
            "visible": False,
            "bbox_xyxy": None,
            "message": NOT_VISIBLE_MESSAGE,
            "label": None,
            "segment_id": segment_id,
        }
    return {
        "has_selection": True,
        "visible": True,
        "bbox_xyxy": bbox,
        "message": None,
        "label": SELECTED_LABEL,
        "segment_id": segment_id,
    }


def selection_summary_card(selection: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not selection:
        return None
    return {
        "candidate_id": selection.get("selected_candidate_id"),
        "segment_id": selection.get("selected_segment_id"),
        "raw_track_id": selection.get("selected_raw_track_id"),
        "selected_frame": selection.get("selected_frame_index"),
        "listed_selection": bool(selection.get("listed_selection")),
        "direct_bbox_selection": bool(selection.get("direct_bbox_selection")),
    }


def confirmation_user_summary(
    *,
    action: str,
    selection: Mapping[str, Any] | None,
    confidence: str,
) -> dict[str, Any]:
    card = selection_summary_card(selection) or {}
    return {
        "seçilen_oyuncu": card.get("candidate_id") or card.get("segment_id"),
        "segment": card.get("segment_id"),
        "raw_track": card.get("raw_track_id"),
        "frame": card.get("selected_frame"),
        "karar": action,
        "confidence": confidence,
        "decision_log_a_yazilacak": True,
        "tracking_calistirilmayacak": True,
        "listed_selection": card.get("listed_selection"),
        "direct_bbox_selection": card.get("direct_bbox_selection"),
    }


def action_requires_selection(action_label: str) -> bool:
    return action_label in {"Confirm Target", "Reject Candidate"}


def build_selection_from_bbox_hit(
    *,
    frame_index: int,
    bbox_xyxy: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
    hit_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from football_analytics.reid.hil_ui.candidates_view import find_candidate_covering_bbox

    listed = find_candidate_covering_bbox(
        candidates, frame_index=frame_index, bbox_xyxy=bbox_xyxy
    )
    meta = dict(hit_meta or {})
    if listed is not None:
        return {
            "selected_candidate_id": listed["candidate_id"],
            "selected_segment_id": listed["segment_id"],
            "selected_raw_track_id": listed["raw_track_id"],
            "selected_frame_index": frame_index,
            "selected_bbox_xyxy": [float(v) for v in bbox_xyxy],
            "direct_bbox_selection": False,
            "listed_selection": True,
            "displayed_rank": listed.get("appearance_rank"),
            "displayed_score": listed.get("S"),
            "displayed_T_max": listed.get("T_max"),
            "displayed_D_max": listed.get("D_max"),
            "displayed_model_id": listed.get("sportsreid_model_id"),
            "displayed_checkpoint_sha256": listed.get("sportsreid_checkpoint_sha256"),
        }
    return {
        "selected_candidate_id": None,
        "selected_segment_id": meta.get("segment_id") or meta.get("bbox_id"),
        "selected_raw_track_id": meta.get("raw_track_id"),
        "selected_frame_index": frame_index,
        "selected_bbox_xyxy": [float(v) for v in bbox_xyxy],
        "direct_bbox_selection": True,
        "listed_selection": False,
        "displayed_rank": None,
        "displayed_score": None,
        "displayed_T_max": None,
        "displayed_D_max": None,
        "displayed_model_id": None,
        "displayed_checkpoint_sha256": None,
    }
