"""Target Tracking R1 — persistent target state + conservative local stitching."""

from football_analytics.reid.target_tracking_r1.policy import STITCH_POLICY
from football_analytics.reid.target_tracking_r1.state import (
    STATE_SCHEMA,
    apply_human_seed,
    empty_persistent_state,
)

__all__ = [
    "STITCH_POLICY",
    "STATE_SCHEMA",
    "apply_human_seed",
    "empty_persistent_state",
]
