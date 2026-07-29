#!/usr/bin/env python3
"""Bootstrap Target Golden Clip R1 readiness artifacts (no detection/tracking rerun).

Writes metric-validity correction + UI readiness report. Does not mutate product logs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.golden_clip.metric_validity import (  # noqa: E402
    build_metric_validity_manifest,
)
from football_analytics.reid.golden_clip.metrics import (  # noqa: E402
    evaluate_variant_against_gt,
    metrics_without_gt,
    select_best_variant,
)
from football_analytics.reid.golden_clip import NOT_MEASURABLE_WITHOUT_GT  # noqa: E402

ROOT = (
    PROJECT
    / "outputs/reid/target_golden_clip_r1/match_short_video_f2f6d8a077ca/sv_run_20260727T234854Z"
)
RUN = (
    PROJECT
    / "outputs/reid/product_new_short_video_preprocess_validation/sv_run_20260727T234854Z"
)
EXPECTED_SHA = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"
EXPECTED_HEAD = "01ce9d7813905b0c5860d317bdd9770271158d99"

VARIANT_ALIASES = {
    "A_current_bytetrack": "A_current_bytetrack",
    "B1_tuned": "B1_buffer60_match07_new035",
    "B2_aggressive": "B2_buffer90_match065_new04",
    "B3_mid": "B3_buffer45_match075_new03",
    "C_botsort_gmc": "C_botsort_gmc_sparseOptFlow",
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    import subprocess

    # Contract checks
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if head != EXPECTED_HEAD:
        print("BLOCKED_TARGET_GOLDEN_CLIP_SOURCE_CONTRACT")
        print(f"HEAD mismatch: {head} != {EXPECTED_HEAD}")
        return 2
    video = PROJECT / "data/product_inputs/short_video_f2f6d8a077ca/kisa_mac_klip.mp4"
    from football_analytics.ingest.checksum import sha256_file

    if sha256_file(video) != EXPECTED_SHA:
        print("BLOCKED_TARGET_GOLDEN_CLIP_SOURCE_CONTRACT")
        print("source video SHA mismatch")
        return 3
    if not RUN.is_dir():
        print("BLOCKED_TARGET_GOLDEN_CLIP_TRACKING_ARTIFACTS")
        return 4

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "session").mkdir(exist_ok=True)
    (ROOT / "metrics").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "overlays").mkdir(exist_ok=True)

    stab = RUN / "tracking_stabilization" / "final_manifest.json"
    prior = json.loads(stab.read_text(encoding="utf-8")) if stab.is_file() else {}
    accepted_path = ROOT / "session" / "accepted_ground_truth.json"
    accepted_gt = accepted_path.is_file() and json.loads(accepted_path.read_text()).get("accepted")

    validity = build_metric_validity_manifest(
        prior_stabilization=prior, accepted_gt=bool(accepted_gt)
    )
    _write(ROOT / "metrics" / "metric_validity_correction.json", validity)

    # Pre-acceptance metrics: NOT_MEASURABLE
    variant_status = {}
    results = []
    for alias, disk_id in VARIANT_ALIASES.items():
        vdir = RUN / "tracking_stabilization" / "variants" / disk_id
        tracks = vdir / "tracks.jsonl"
        if not tracks.is_file():
            variant_status[alias] = "VARIANT_ARTIFACT_NOT_PRESENT"
            results.append(
                {
                    "variant_id": alias,
                    "status": "VARIANT_ARTIFACT_NOT_PRESENT",
                    "false_target_identity_switch_count": NOT_MEASURABLE_WITHOUT_GT,
                }
            )
            continue
        variant_status[alias] = "PRESENT"
        summary = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
        probe = (summary.get("continuity_probe") or {}).get("uninterrupted_duration_sec")
        if not accepted_gt:
            row = metrics_without_gt()
            row["variant_id"] = alias
            row["runtime_sec"] = summary.get("runtime_sec")
            row["seed_iou_continuity_proxy_seconds"] = probe
            row["disk_variant_id"] = disk_id
            results.append(row)
        else:
            # Lazy load only when accepted
            obs = [
                json.loads(line)
                for line in tracks.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            gt = json.loads(accepted_path.read_text(encoding="utf-8"))
            row = evaluate_variant_against_gt(
                ground_truth=gt,
                variant_observations=obs,
                fps=30.0,
                variant_id=alias,
                runtime_sec=summary.get("runtime_sec"),
                seed_iou_continuity_proxy_seconds=probe,
            )
            row["disk_variant_id"] = disk_id
            results.append(row)

    selection = select_best_variant([r for r in results if r.get("status") == "ok"])
    _write(
        ROOT / "metrics" / "variant_evaluation_pre_acceptance.json",
        {
            "created_at": _utc(),
            "accepted_gt": bool(accepted_gt),
            "variant_artifact_status": variant_status,
            "results": results,
            "selection": selection if accepted_gt else None,
            "note_tr": (
                "Accepted GT yokken false_target_identity_switch = "
                f"{NOT_MEASURABLE_WITHOUT_GT}. "
                "seed_iou_continuity_proxy_seconds hedef doğruluğu değildir."
            ),
        },
    )

    status = (
        "COMPLETED_TARGET_GOLDEN_CLIP_ANNOTATION_AND_METRIC_BASELINE"
        if accepted_gt
        else "COMPLETED_TARGET_GOLDEN_CLIP_UI_READY_USER_ACTION_REQUIRED"
    )
    report = {
        "final_status": status,
        "created_at": _utc(),
        "match_id": "match_short_video_f2f6d8a077ca",
        "analysis_run_id": "sv_run_20260727T234854Z",
        "target_id": "target_001",
        "source_video_sha256": EXPECTED_SHA,
        "head_sha": head,
        "detection_rerun": False,
        "tracking_rerun": False,
        "product_logs_mutated": False,
        "variant_artifact_status": variant_status,
        "metric_validity_path": str(ROOT / "metrics" / "metric_validity_correction.json"),
        "golden_root": str(ROOT),
        "next_user_action": "Open Target Ground Truth UI and annotate intervals",
        "forbidden_next": ["HIL-D", "Game State", "CLIP", "2D", "heatmap", "new tracker"],
    }
    _write(ROOT / "reports" / "final_manifest.json", report)
    (ROOT / "reports" / "final_report.md").write_text(
        "\n".join(
            [
                "# Target Golden Clip R1 — Final Report",
                "",
                f"- final_status: `{status}`",
                f"- match_id: `match_short_video_f2f6d8a077ca`",
                f"- analysis_run_id: `sv_run_20260727T234854Z`",
                f"- source_sha: `{EXPECTED_SHA}`",
                "",
                "## Metrik geçerliliği",
                "",
                f"- Eski `false_target_identity_switch=0` placeholder’dır; accepted GT yokken "
                f"`{NOT_MEASURABLE_WITHOUT_GT}` olarak raporlanır.",
                "- Eski `continuity_probe≈25.3s` artık `seed_iou_continuity_proxy_seconds` "
                "olarak adlandırılır ve hedef doğruluğu değildir.",
                "",
                "## Kullanıcı aksiyonu",
                "",
                "- Target Ground Truth UI’de interval/track-assisted annotation yapın.",
                "- Overlay’i izleyip acceptance verin.",
                "- Acceptance sonrası metrik baseline script’i tekrar çalıştırılmalıdır.",
                "",
                "## Yasak sonraki adımlar",
                "",
                "- HIL-D / Game State / 2D / heatmap / yeni tracker",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(status)
    print(f"golden_root={ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
