"""Candidate presentation helpers — ordering never shrinks the universe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

FORBIDDEN_SCORE_LABELS = {
    "probability",
    "confidence percentage",
    "identity probability",
    "prob",
    "p_target",
}


def eligible_candidate_universe(
    candidates: Sequence[Mapping[str, Any]],
    *,
    eligible_only: bool = False,
) -> list[dict[str, Any]]:
    """Return full candidate list (copy). Never top-k truncates."""
    rows = [dict(c) for c in candidates]
    if eligible_only:
        rows = [c for c in rows if c.get("eligibility") is True]
    return rows


def sort_candidates_for_display(
    candidates: Sequence[Mapping[str, Any]],
    *,
    order: str = "display_order",
) -> list[dict[str, Any]]:
    """Sort for presentation; universe membership unchanged."""
    rows = eligible_candidate_universe(candidates, eligible_only=False)
    if order == "appearance_rank":
        rows.sort(
            key=lambda c: (
                c.get("appearance_rank") is None,
                c.get("appearance_rank") if c.get("appearance_rank") is not None else 10**9,
                c.get("display_order", 10**9),
                c.get("candidate_id", ""),
            )
        )
    elif order == "display_order":
        rows.sort(
            key=lambda c: (c.get("display_order", 10**9), c.get("candidate_id", ""))
        )
    else:
        raise ValueError(f"unsupported display order: {order!r}")
    return rows


def assert_no_topk_hiding(
    original: Sequence[Mapping[str, Any]],
    displayed: Sequence[Mapping[str, Any]],
) -> None:
    orig_ids = {c["candidate_id"] for c in original}
    disp_ids = {c["candidate_id"] for c in displayed}
    if orig_ids != disp_ids:
        missing = sorted(orig_ids - disp_ids)
        raise AssertionError(f"candidate universe shrunk; missing={missing}")


def sportsreid_display_fields(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Labels for helper scores — never probability language."""
    return {
        "appearance_rank": candidate.get("appearance_rank"),
        "similarity_T_max": candidate.get("T_max"),
        "similarity_D_max": candidate.get("D_max"),
        "similarity_margin_S": candidate.get("S"),
        "score_semantics": candidate.get(
            "score_semantics", "similarity_margin_not_probability"
        ),
        "forbidden_labels_rejected": sorted(FORBIDDEN_SCORE_LABELS),
        "warning": "SportsReID is a helper ranker; it does not confirm identity.",
    }


def find_candidate_covering_bbox(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frame_index: int,
    bbox_xyxy: Sequence[float],
    atol: float = 1e-3,
) -> dict[str, Any] | None:
    """Match a click selection to a listed candidate bbox if present."""
    tx = [float(v) for v in bbox_xyxy]
    for cand in candidates:
        for ref in cand.get("bbox_references") or []:
            if int(ref["frame_index"]) != int(frame_index):
                continue
            bx = [float(v) for v in ref["bbox_xyxy"]]
            if all(abs(a - b) <= atol for a, b in zip(tx, bx)):
                return dict(cand)
    return None
