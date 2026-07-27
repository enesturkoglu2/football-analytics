"""Isolated identity for short-video product runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


def new_analysis_run_id(prefix: str = "sv_run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


@dataclass(frozen=True)
class ShortVideoIdentity:
    match_id: str
    analysis_run_id: str
    video_id: str
    video_sha256: str
    target_id: str
    product_package_id: str

    @property
    def composite_key(self) -> str:
        return f"{self.match_id}+{self.analysis_run_id}+{self.target_id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["composite_key"] = self.composite_key
        return d


def build_identity(*, video_sha256: str, analysis_run_id: str | None = None) -> ShortVideoIdentity:
    run_id = analysis_run_id or new_analysis_run_id()
    short = video_sha256[:12]
    return ShortVideoIdentity(
        match_id=f"match_short_video_{short}",
        analysis_run_id=run_id,
        video_id=f"video_short_{short}",
        video_sha256=video_sha256,
        target_id="target_001",
        product_package_id=f"pkg_short_video_{short}_{run_id}",
    )
