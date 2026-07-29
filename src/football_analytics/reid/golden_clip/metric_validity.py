"""Metric-validity correction for prior placeholder / proxy metrics (no overwrite)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from football_analytics.reid.golden_clip import NOT_MEASURABLE_WITHOUT_GT


def build_metric_validity_manifest(
    *,
    prior_stabilization: Mapping[str, Any] | None,
    accepted_gt: bool,
) -> dict[str, Any]:
    """Correct naming/validity without mutating old stabilization artifacts."""
    prior = prior_stabilization or {}
    probe = None
    # Prefer nested continuity_probe from final_manifest baseline_summary
    for key in ("baseline_summary", "product_candidate_summary", "target_diagnostic"):
        block = prior.get(key) or {}
        cp = block.get("continuity_probe") or block.get("continuity_probe_on_longest_track")
        if isinstance(cp, Mapping) and cp.get("uninterrupted_duration_sec") is not None:
            probe = float(cp["uninterrupted_duration_sec"])
            break
    if probe is None and prior.get("false_target_identity_switch") is not None:
        # still record correction even if probe missing
        pass

    return {
        "schema_version": "target_golden_clip_metric_validity_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prior_artifact_mutated": False,
        "corrections": [
            {
                "old_name": "false_target_identity_switch",
                "old_value": prior.get("false_target_identity_switch", 0),
                "old_interpretation": "placeholder_constant_zero_without_gt",
                "new_name": "false_target_identity_switch_count",
                "new_value_without_accepted_gt": NOT_MEASURABLE_WITHOUT_GT,
                "new_value_with_accepted_gt": (
                    "computed_from_target_identity_ground_truth_v1"
                    if accepted_gt
                    else NOT_MEASURABLE_WITHOUT_GT
                ),
                "must_not_treat_old_zero_as_measured": True,
            },
            {
                "old_name": "continuity_probe",
                "old_value_seconds": probe,
                "new_name": "seed_iou_continuity_proxy_seconds",
                "new_value_seconds": probe,
                "is_target_accuracy": False,
                "note_tr": (
                    "seed_iou_continuity_proxy_seconds yalnızca seed bbox üzerinden "
                    "greedy IoU süreklilik proxy’sidir; hedef kimliği doğruluğunu ölçmez."
                ),
            },
        ],
        "accepted_gt_present": bool(accepted_gt),
    }
