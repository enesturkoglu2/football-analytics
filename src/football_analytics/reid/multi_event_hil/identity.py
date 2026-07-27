"""Isolated match / analysis / target identity contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


TARGET_ID = "target_001"
VIDEO_ID = "target_001_external_enrollment_v1"
VIDEO_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"
MATCH_ID = "match_external_enrollment_v1_mehil"
PACKAGE_ID = "pkg_mehil_external_enrollment_v1"


def new_analysis_run_id(*, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"mehil_run_{stamp}"


@dataclass(frozen=True)
class AnalysisIdentity:
    match_id: str
    analysis_run_id: str
    target_id: str
    product_package_id: str
    video_id: str
    video_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def composite_key(self) -> str:
        return f"{self.match_id}+{self.analysis_run_id}+{self.target_id}"


def build_identity(analysis_run_id: str | None = None) -> AnalysisIdentity:
    return AnalysisIdentity(
        match_id=MATCH_ID,
        analysis_run_id=analysis_run_id or new_analysis_run_id(),
        target_id=TARGET_ID,
        product_package_id=PACKAGE_ID,
        video_id=VIDEO_ID,
        video_sha256=VIDEO_SHA,
    )
