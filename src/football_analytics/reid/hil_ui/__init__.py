"""HIL-B offline review UI domain logic (Streamlit-independent where possible)."""

from __future__ import annotations

from football_analytics.reid.hil_ui.actions import (
    UI_ACTION_TO_DECISION,
    ConfirmationPreview,
    map_ui_action,
    preview_confirmation,
)
from football_analytics.reid.hil_ui.geometry import (
    BBoxHit,
    ClickResolution,
    letterbox_params,
    resolve_click_to_bbox,
    ui_to_frame_coords,
)
from football_analytics.reid.hil_ui.package import (
    REVIEW_PACKAGE_SCHEMA_VERSION,
    ReviewPackageError,
    load_and_validate_review_package,
    validate_review_package,
)

__all__ = [
    "BBoxHit",
    "ClickResolution",
    "ConfirmationPreview",
    "REVIEW_PACKAGE_SCHEMA_VERSION",
    "ReviewPackageError",
    "UI_ACTION_TO_DECISION",
    "letterbox_params",
    "load_and_validate_review_package",
    "map_ui_action",
    "preview_confirmation",
    "resolve_click_to_bbox",
    "ui_to_frame_coords",
    "validate_review_package",
]
