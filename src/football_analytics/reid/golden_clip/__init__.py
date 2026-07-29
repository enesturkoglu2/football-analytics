"""Target Golden Clip R1 — human identity ground truth + tracking metrics."""

from __future__ import annotations

SCHEMA_GT = "target_identity_ground_truth_v1"
SCHEMA_ANNOTATION_EVENT = "target_gt_annotation_event_v1"
UI_PROFILE = "target_ground_truth"
PACKAGE_MODE = "golden_clip_annotation"

TARGET_STATES = frozenset(
    {
        "TARGET_VISIBLE_ASSOCIATED",
        "TARGET_VISIBLE_BUT_MISSED",
        "TARGET_OCCLUDED",
        "TARGET_OUT_OF_FRAME",
        "TARGET_UNCERTAIN",
        "WRONG_TARGET_ASSIGNED",
    }
)

VISIBLE_STATES = frozenset(
    {
        "TARGET_VISIBLE_ASSOCIATED",
        "TARGET_VISIBLE_BUT_MISSED",
        "WRONG_TARGET_ASSIGNED",
    }
)

CORRECT_ASSIGNMENT_STATES = frozenset({"TARGET_VISIBLE_ASSOCIATED"})

NOT_MEASURABLE_WITHOUT_GT = "NOT_MEASURABLE_WITHOUT_GT"
