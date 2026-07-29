#!/usr/bin/env python3
"""Evaluate accepted GT against existing versioned tracking variants (no rerun)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.golden_clip.error_taxonomy import (  # noqa: E402
    choose_next_gate,
    classify_failures,
)
from football_analytics.reid.golden_clip.metrics import (  # noqa: E402
    evaluate_variant_against_gt,
    select_best_variant,
)
from football_analytics.reid.golden_clip.overlay import render_gt_overlay_video  # noqa: E402
from football_analytics.reid.golden_clip.validate import validate_ground_truth  # noqa: E402
from football_analytics.reid.golden_clip import NOT_MEASURABLE_WITHOUT_GT  # noqa: E402

ROOT = (
    PROJECT
    / "outputs/reid/target_golden_clip_r1/match_short_video_f2f6d8a077ca/sv_run_20260727T234854Z"
)
RUN = (
    PROJECT
    / "outputs/reid/product_new_short_video_preprocess_validation/sv_run_20260727T234854Z"
)
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


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    accepted_path = ROOT / "session" / "accepted_ground_truth.json"
    if not accepted_path.is_file():
        print("NO_ACCEPTED_GT")
        print(f"false_target_identity_switch={NOT_MEASURABLE_WITHOUT_GT}")
        return 2
    gt = json.loads(accepted_path.read_text(encoding="utf-8"))
    if not gt.get("accepted"):
        print("NO_ACCEPTED_GT")
        return 3

    refs = json.loads((ROOT / "read_only_refs" / "artifact_pointers.json").read_text())
    report = validate_ground_truth(
        gt,
        expected_source_sha256=refs["source_video_sha256"],
        frame_count=int(refs["frame_count"]),
        fps=float(refs["fps"]),
    )
    _write(ROOT / "session" / "acceptance_validation_report.json", report)
    if not report["ok"]:
        print("FAILED_TARGET_GOLDEN_CLIP_ANNOTATION_CONSISTENCY")
        return 4

    video = PROJECT / refs["source_video"]
    overlay = render_gt_overlay_video(
        video_path=video,
        ground_truth=gt,
        output_path=ROOT / "overlays" / "gt_review_overlay.mp4",
    )
    _write(ROOT / "overlays" / "gt_overlay_meta.json", overlay)

    dets = _load_jsonl(Path(refs["detection_jsonl"]))
    results = []
    taxonomies = []
    for alias, disk_id in VARIANT_ALIASES.items():
        vdir = RUN / "tracking_stabilization" / "variants" / disk_id
        tracks = vdir / "tracks.jsonl"
        if not tracks.is_file():
            results.append(
                {
                    "variant_id": alias,
                    "status": "VARIANT_ARTIFACT_NOT_PRESENT",
                    "false_target_identity_switch_count": NOT_MEASURABLE_WITHOUT_GT,
                }
            )
            continue
        obs = _load_jsonl(tracks)
        summary = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
        probe = (summary.get("continuity_probe") or {}).get("uninterrupted_duration_sec")
        row = evaluate_variant_against_gt(
            ground_truth=gt,
            variant_observations=obs,
            fps=float(refs["fps"]),
            variant_id=alias,
            runtime_sec=summary.get("runtime_sec"),
            seed_iou_continuity_proxy_seconds=probe,
        )
        results.append(row)
        tax = classify_failures(
            ground_truth=gt,
            variant_observations=obs,
            detections=dets,
            fps=float(refs["fps"]),
            variant_id=alias,
        )
        taxonomies.append(tax)

    selection = select_best_variant([r for r in results if r.get("status") == "ok"])
    gate = choose_next_gate(taxonomies)
    payload = {
        "created_at": _utc(),
        "accepted_gt": True,
        "results": results,
        "selection": selection,
        "error_taxonomy": taxonomies,
        "exact_next_gate": gate,
        "metric_definitions_tr": {
            "target_visible_duration": "VISIBLE_STATES frame sayısı / fps",
            "correctly_assigned_duration": "IoU>=0.5 ve doğru association frame / fps",
            "target_precision": "correct / claim_frames",
            "target_recall": "correct / visible_frames (UNCERTAIN hariç)",
            "target_association_f1": "2PR/(P+R) tek-hedef IDF1 proxy",
            "seed_iou_continuity_proxy_seconds": "GT değildir; eski continuity_probe",
        },
    }
    _write(ROOT / "metrics" / "variant_evaluation_accepted.json", payload)
    _write(ROOT / "reports" / "exact_next_gate.json", gate)

    status = "COMPLETED_TARGET_GOLDEN_CLIP_ANNOTATION_AND_METRIC_BASELINE"
    _write(
        ROOT / "reports" / "final_manifest.json",
        {
            "final_status": status,
            "created_at": _utc(),
            "exact_next_gate": gate.get("exact_next_gate"),
            "selection": selection,
            "overlay": overlay,
        },
    )
    print(status)
    print(f"exact_next_gate={gate.get('exact_next_gate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
