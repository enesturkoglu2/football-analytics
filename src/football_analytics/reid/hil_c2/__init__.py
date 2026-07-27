"""HIL-C2 package exports."""

from __future__ import annotations

from football_analytics.reid.hil_c2.product_package import (
    PRODUCT_PACKAGE_ID,
    PRODUCT_RUN_ID,
    VIDEO_SHA,
    build_product_external_review_package,
)
from football_analytics.reid.hil_c2.source_audit import (
    ProductSourceClass,
    audit_product_video_sources,
)

__all__ = [
    "PRODUCT_PACKAGE_ID",
    "PRODUCT_RUN_ID",
    "PRODUCT_VIDEO_SHA",
    "ProductSourceClass",
    "VIDEO_SHA",
    "audit_product_video_sources",
    "build_product_external_review_package",
]

PRODUCT_VIDEO_SHA = VIDEO_SHA
