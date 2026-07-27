"""UI action mapping and confirmation preview (Streamlit-independent)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from football_analytics.reid.hil.decisions import DecisionAction

UI_ACTION_TO_DECISION: dict[str, DecisionAction] = {
    "Confirm Target": DecisionAction.CONFIRM_TARGET,
    "Reject Candidate": DecisionAction.REJECT_CANDIDATE,
    "None of These": DecisionAction.NONE_OF_THESE,
    "Unknown": DecisionAction.UNKNOWN,
    "Invalid Segment": DecisionAction.INVALID_SEGMENT,
    "Defer": DecisionAction.DEFER,
    "Undo Last Decision": DecisionAction.CORRECT_PREVIOUS_DECISION,  # undo via supersede path
    "Correct Previous Decision": DecisionAction.CORRECT_PREVIOUS_DECISION,
    "Revoke Decision": DecisionAction.REVOKE,
}


@dataclass(frozen=True)
class ConfirmationPreview:
    action: str
    selected_candidate_id: str | None
    selected_segment_id: str | None
    selected_raw_track_id: str | None
    selected_frame_index: int | None
    selected_bbox_xyxy: list[float] | None
    direct_bbox_selection: bool
    listed_selection: bool
    displayed_rank: int | None
    displayed_score: float | None
    displayed_T_max: float | None
    displayed_D_max: float | None
    score_label: str
    comment: str
    reviewer: str
    confidence: str
    training_use_approved: bool
    gallery_use_approved: bool
    helper_warning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_ui_action(label: str) -> DecisionAction:
    try:
        return UI_ACTION_TO_DECISION[label]
    except KeyError as exc:
        raise ValueError(f"unknown UI action label: {label!r}") from exc


def preview_confirmation(
    *,
    action: str | DecisionAction,
    selection: Mapping[str, Any] | None,
    reviewer: str,
    comment: str = "",
    confidence: str = "unknown",
    training_use_approved: bool = False,
    gallery_use_approved: bool = False,
) -> ConfirmationPreview:
    """Build confirmation panel payload; approvals default false and never auto-true."""
    action_value = action.value if isinstance(action, DecisionAction) else str(action)
    sel = dict(selection or {})
    direct = bool(sel.get("direct_bbox_selection", False))
    listed = bool(sel.get("listed_selection", not direct and sel.get("selected_candidate_id")))
    if training_use_approved is not True:
        training_use_approved = False
    if gallery_use_approved is not True:
        gallery_use_approved = False
    return ConfirmationPreview(
        action=action_value,
        selected_candidate_id=sel.get("selected_candidate_id"),
        selected_segment_id=sel.get("selected_segment_id"),
        selected_raw_track_id=sel.get("selected_raw_track_id"),
        selected_frame_index=sel.get("selected_frame_index"),
        selected_bbox_xyxy=list(sel["selected_bbox_xyxy"])
        if sel.get("selected_bbox_xyxy") is not None
        else None,
        direct_bbox_selection=direct,
        listed_selection=listed,
        displayed_rank=sel.get("displayed_rank"),
        displayed_score=sel.get("displayed_score"),
        displayed_T_max=sel.get("displayed_T_max"),
        displayed_D_max=sel.get("displayed_D_max"),
        score_label="appearance_similarity_helper",
        comment=comment,
        reviewer=reviewer,
        confidence=confidence,
        training_use_approved=training_use_approved,
        gallery_use_approved=gallery_use_approved,
        helper_warning="SportsReID is a helper ranker; it does not confirm identity.",
    )


NAVIGATION_ACTIONS = frozenset({"Previous Event", "Next Event"})
