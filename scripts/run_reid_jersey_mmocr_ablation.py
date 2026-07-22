#!/usr/bin/env python
"""Stage 5C-C2 controlled recognizer/preprocessing ablation runner.

Runs the fixed four-variant matrix (direct SAR at 1x/2x/4x INTER_CUBIC and
DBNet+SAR at 4x) over the same 46 frozen C1 crops. Phases mirror the C1
runner so the model process never sees manual labels:

- ``prepare``  : verify the C1 baseline freeze and copy its blind inference
                 manifest and evaluation reference byte-identically into the
                 temp output dir (no new selection).
- ``infer``    : the model process. Reads ONLY the blind manifest and config;
                 initializes DBNet and SAR once each from validated local
                 paths; runs 184 predictions; writes
                 ``ablation_predictions.jsonl``. Wrap with strace for the
                 network audit.
- ``evaluate`` : joins predictions with the evaluation reference; writes item
                 evaluation, variant summary, and comparison summary.
- ``finalize`` : classifies the strace log (loopback-only policy), writes the
                 network audit, runtime summary, and run manifest; validates
                 the output contract; atomically renames temp to final.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid import jersey_mmocr as jm

BLIND_MANIFEST_NAME = "ablation_inference_manifest.jsonl"
REFERENCE_NAME = "ablation_evaluation_reference.jsonl"
PREDICTIONS_NAME = "ablation_predictions.jsonl"
ITEM_EVALUATION_NAME = "ablation_item_evaluation.jsonl"
VARIANT_SUMMARY_NAME = "ablation_variant_summary.json"
COMPARISON_SUMMARY_NAME = "ablation_comparison_summary.json"
RUNTIME_SUMMARY_NAME = "ablation_runtime_summary.json"
RUN_MANIFEST_NAME = "ablation_run_manifest.json"
NETWORK_AUDIT_NAME = "ablation_network_audit.txt"
INFER_META_NAME = "_infer_meta.json"

FINAL_ARTIFACTS = (
    BLIND_MANIFEST_NAME,
    REFERENCE_NAME,
    PREDICTIONS_NAME,
    ITEM_EVALUATION_NAME,
    VARIANT_SUMMARY_NAME,
    COMPARISON_SUMMARY_NAME,
    RUNTIME_SUMMARY_NAME,
    RUN_MANIFEST_NAME,
    NETWORK_AUDIT_NAME,
)

WARNING_PATTERNS = {
    "state_dict_unexpected_key_data_preprocessor": re.compile(
        r"unexpected key in source state_dict:.*data_preprocessor\.mean, data_preprocessor\.std"
    ),
    "state_dict_mismatch_notice": re.compile(
        r"The model and loaded state dict do not match exactly"
    ),
    "mmengine_registry_scope_warning": re.compile(
        r'Failed to search registry with scope "mmocr"'
    ),
    "local_vis_backend_warning": re.compile(r"LocalVisBackend"),
}


def _fail(message: str) -> None:
    raise jm.JerseyMMOCRError(message)


def _expected_counts(config: dict) -> dict[str, int]:
    return {str(key): int(value) for key, value in config["selection"]["expected_counts"].items()}


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = jm.sha256_file(path)
    if actual != expected:
        _fail(f"{label} sha256 mismatch: expected {expected}, got {actual}")


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["device"] != "cpu":
        _fail("only device=cpu is allowed")
    if not config.get("offline_required"):
        _fail("offline_required must be true")
    if config["digit_policy"].get("confidence_threshold") is not None:
        _fail("confidence_threshold must be null")
    if config.get("interpolation") != "INTER_CUBIC":
        _fail("interpolation must be INTER_CUBIC")
    jm.validate_ablation_variants(config["variants"])
    return config


def phase_prepare(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    freeze = _PROJECT_ROOT / config["baseline_freeze"]["path"]
    blind_src = freeze / "smoke_inference_manifest.jsonl"
    ref_src = freeze / "smoke_evaluation_reference.jsonl"
    _verify_sha(blind_src, config["baseline_freeze"]["inference_manifest_sha256"], "C1 blind manifest")
    _verify_sha(ref_src, config["baseline_freeze"]["evaluation_reference_sha256"], "C1 evaluation reference")
    _verify_sha(
        freeze / "baseline_freeze_summary.json",
        config["baseline_freeze"]["freeze_summary_sha256"],
        "C1 freeze summary",
    )
    _verify_sha(
        freeze / "baseline_freeze_manifest.json",
        config["baseline_freeze"]["freeze_manifest_sha256"],
        "C1 freeze manifest",
    )
    shutil.copyfile(blind_src, output_dir / BLIND_MANIFEST_NAME)
    shutil.copyfile(ref_src, output_dir / REFERENCE_NAME)
    if (output_dir / BLIND_MANIFEST_NAME).read_bytes() != blind_src.read_bytes():
        _fail("blind manifest copy is not byte-identical")
    if (output_dir / REFERENCE_NAME).read_bytes() != ref_src.read_bytes():
        _fail("evaluation reference copy is not byte-identical")

    blind = jm.load_jsonl(output_dir / BLIND_MANIFEST_NAME)
    refs = jm.load_jsonl(output_dir / REFERENCE_NAME)
    jm.validate_manifest_pair(blind, refs, int(config["max_items"]))
    print(f"PREPARE_OK items={len(blind)}", flush=True)


def phase_infer(args: argparse.Namespace, config: dict) -> None:
    if args.device != "cpu":
        _fail("only --device cpu is allowed")
    if not args.offline_required:
        _fail("--offline-required must be set")
    output_dir = Path(args.output_dir)

    _verify_sha(Path(args.asset_manifest), config["asset_manifest_sha256"], "asset_manifest")
    _verify_sha(
        Path(config["config_closure_path"]), config["config_closure_sha256"], "config_closure"
    )

    # The model process reads ONLY the blind manifest; the reference stays closed.
    blind = jm.load_jsonl(Path(args.inference_manifest))
    jm.assert_blind_records_safe(blind)
    if len(blind) != int(config["max_items"]):
        _fail(f"blind manifest must have {config['max_items']} rows, got {len(blind)}")

    import torch

    if torch.cuda.is_available():
        _fail("CUDA must not be available")
    torch.manual_seed(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector, recognizer, init_timings = jm.build_inferencers(config)
    python_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    model_meta = jm.build_model_meta(config)
    start = time.perf_counter()
    predictions = jm.run_ablation_inference(
        blind, config["variants"], detector, recognizer, model_meta
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    if len(predictions) != int(config["expected_prediction_count"]):
        _fail(
            f"expected {config['expected_prediction_count']} predictions, got {len(predictions)}"
        )
    jm.write_jsonl(output_dir / PREDICTIONS_NAME, predictions)

    peak_rss_kb = None
    try:
        import resource

        peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        pass

    import importlib.metadata as importlib_metadata

    environment = {
        "python_version": platform.python_version(),
        "packages": {
            name: importlib_metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "mmcv",
                "mmengine",
                "mmdet",
                "mmocr",
            )
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
        "cpu": platform.processor() or platform.machine(),
    }
    jm.write_json(
        output_dir / INFER_META_NAME,
        {
            "init_timings": init_timings,
            "inference_wall_ms": wall_ms,
            "peak_rss_kb": peak_rss_kb,
            "environment": environment,
            "python_warnings": python_warnings,
            "prediction_count": len(predictions),
        },
    )
    print(f"INFER_OK predictions={len(predictions)}", flush=True)


def phase_evaluate(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    predictions = jm.load_jsonl(output_dir / PREDICTIONS_NAME)
    # Predictions are complete on disk; only now is the reference opened.
    reference_rows = jm.load_jsonl(Path(args.evaluation_reference))
    item_rows, variant_summary = jm.evaluate_ablation_predictions(
        predictions, reference_rows, _expected_counts(config)
    )
    comparison = jm.build_ablation_comparison_summary(
        variant_summary, config["c1_baseline_metrics"]
    )
    jm.write_jsonl(output_dir / ITEM_EVALUATION_NAME, item_rows)
    jm.write_json(output_dir / VARIANT_SUMMARY_NAME, variant_summary)
    jm.write_json(output_dir / COMPARISON_SUMMARY_NAME, comparison)
    print(f"EVALUATE_OK items={len(item_rows)}", flush=True)


def _count_log_warnings(log_text: str) -> dict[str, int]:
    return {name: len(pattern.findall(log_text)) for name, pattern in WARNING_PATTERNS.items()}


def phase_finalize(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    final_dir = _PROJECT_ROOT / config["output"]["final_dir"]
    if final_dir.exists():
        _fail(f"final output already exists (no overwrite): {final_dir}")

    strace_text = Path(args.strace_log).read_text(encoding="utf-8", errors="replace")
    audit = jm.parse_network_strace(strace_text)
    if audit["policy_status"] != "pass_loopback_only":
        _fail(f"network audit failed: {json.dumps(audit)}")

    infer_log_text = ""
    if args.infer_log:
        infer_log_text = Path(args.infer_log).read_text(encoding="utf-8", errors="replace")
    log_warning_counts = _count_log_warnings(infer_log_text)

    infer_meta = json.loads((output_dir / INFER_META_NAME).read_text(encoding="utf-8"))
    predictions = jm.load_jsonl(output_dir / PREDICTIONS_NAME)
    error_count = sum(1 for p in predictions if p.get("inference_error") is not None)
    overall_status = (
        "completed" if error_count == 0 else "completed_with_inference_errors"
    )

    warning_summary = {
        "python_warnings": infer_meta["python_warnings"],
        "log_warning_counts": log_warning_counts,
        "state_dict_unexpected_keys": {
            "message": (
                "unexpected key in source state_dict: data_preprocessor.mean, "
                "data_preprocessor.std"
            ),
            "occurrence_count": log_warning_counts[
                "state_dict_unexpected_key_data_preprocessor"
            ],
            "fail": False,
        },
    }

    def stats(values):
        import math as _math
        import statistics as _stats

        finite = [float(v) for v in values if v is not None]
        if not finite:
            return None
        ordered = sorted(finite)
        p95_index = max(0, _math.ceil(0.95 * len(ordered)) - 1)
        return {
            "count": len(finite),
            "min_ms": min(finite),
            "max_ms": max(finite),
            "mean_ms": _stats.fmean(finite),
            "median_ms": _stats.median(finite),
            "p95_ms": ordered[p95_index],
            "sum_ms": sum(finite),
        }

    per_variant_runtime = {}
    for variant_id in jm.ABLATION_VARIANT_IDS:
        variant_predictions = [p for p in predictions if p["variant_id"] == variant_id]
        per_variant_runtime[variant_id] = {
            "total": stats([p.get("total_runtime_ms") for p in variant_predictions]),
            "resize": stats([p.get("resize_runtime_ms") for p in variant_predictions]),
            "detector": stats([p.get("detector_runtime_ms") for p in variant_predictions]),
            "recognizer": stats([p.get("recognizer_runtime_ms") for p in variant_predictions]),
        }
    runtime_summary = {
        "schema_version": jm.ABLATION_RUNTIME_SCHEMA_VERSION,
        "overall_status": overall_status,
        "detector_init_ms": infer_meta["init_timings"].get("detector_init_ms"),
        "recognizer_init_ms": infer_meta["init_timings"].get("recognizer_init_ms"),
        "inference_wall_ms": infer_meta["inference_wall_ms"],
        "per_variant": per_variant_runtime,
        "peak_rss_kb": infer_meta["peak_rss_kb"],
        "environment": infer_meta["environment"],
        "warnings": warning_summary,
        "input_color_convention": "bgr",
    }
    jm.write_json(output_dir / RUNTIME_SUMMARY_NAME, runtime_summary)

    audit_lines = [
        f"{key}={json.dumps(value)}" for key, value in audit.items() if key != "violations"
    ]
    audit_lines.append(f"violations_count={len(audit['violations'])}")
    audit_lines.extend(f"violation: {line}" for line in audit["violations"])
    (output_dir / NETWORK_AUDIT_NAME).write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    git_head = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (output_dir / INFER_META_NAME).unlink()

    comparison = json.loads((output_dir / COMPARISON_SUMMARY_NAME).read_text(encoding="utf-8"))

    def row_count(path: Path):
        if path.suffix == ".jsonl":
            return sum(1 for line in path.read_text().splitlines() if line.strip())
        return None

    artifact_hashes = {}
    for name in FINAL_ARTIFACTS:
        if name == RUN_MANIFEST_NAME:
            continue
        path = output_dir / name
        if not path.is_file():
            _fail(f"missing artifact before finalize: {name}")
        artifact_hashes[name] = {
            "sha256": jm.sha256_file(path),
            "byte_size": path.stat().st_size,
            "row_count": row_count(path),
        }

    freeze = _PROJECT_ROOT / config["baseline_freeze"]["path"]
    cache_audit = json.loads(Path(args.cache_audit).read_text(encoding="utf-8")) if args.cache_audit else None
    run_manifest = {
        "schema_version": jm.ABLATION_RUN_MANIFEST_SCHEMA_VERSION,
        "stage": "5C-C2",
        "overall_status": overall_status,
        "created_at": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(),
        "project_git_head": git_head,
        "config_path": str(Path(args.config)),
        "config_sha256": jm.sha256_file(args.config),
        "c1_baseline_freeze": {
            "path": str(freeze),
            "inference_manifest_sha256": config["baseline_freeze"]["inference_manifest_sha256"],
            "evaluation_reference_sha256": config["baseline_freeze"][
                "evaluation_reference_sha256"
            ],
            "freeze_summary_sha256": config["baseline_freeze"]["freeze_summary_sha256"],
            "freeze_manifest_sha256": config["baseline_freeze"]["freeze_manifest_sha256"],
        },
        "environment": infer_meta["environment"],
        "models": jm.build_model_meta(config),
        "asset_manifest_sha256": config["asset_manifest_sha256"],
        "config_closure_sha256": config["config_closure_sha256"],
        "experiment_matrix": [dict(v) for v in config["variants"]],
        "interpolation": "INTER_CUBIC",
        "network_isolation": {
            "unshare_supported": False,
            "methods": [
                "temporary HOME/XDG_CACHE_HOME/TORCH_HOME/TMPDIR",
                "invalid HTTP/HTTPS/ALL proxy",
                "strace -f -yy -s 256 -e trace=network",
            ],
            "policy": "loopback_only",
            "audit": {key: value for key, value in audit.items() if key != "violations"},
            "audit_violation_count": len(audit["violations"]),
        },
        "cache_audit": cache_audit,
        "source_immutability": {
            "c1_freeze_unmodified": True,
            "source_crops_unmodified": True,
            "roi_coordinates_unmodified": True,
        },
        "runtime_summary": {
            "detector_init_ms": runtime_summary["detector_init_ms"],
            "recognizer_init_ms": runtime_summary["recognizer_init_ms"],
            "inference_wall_ms": runtime_summary["inference_wall_ms"],
            "peak_rss_kb": runtime_summary["peak_rss_kb"],
        },
        "evidence_labels": comparison["evidence_labels"],
        "interpretation_limits": comparison["interpretation_limits"],
        "artifacts": artifact_hashes,
        "safety_flags": {
            "external_network_accessed": False,
            "automatic_download_attempted": False,
            "package_installed": False,
            "environment_modified": False,
            "dataset_downloaded": False,
            "new_checkpoint_downloaded": False,
            "source_crop_modified": False,
            "roi_coordinates_modified": False,
            "roi_file_created": False,
            "output_image_created": False,
            "sharpening_applied": False,
            "contrast_enhancement_applied": False,
            "threshold_selected": False,
            "model_trained": False,
            "model_finetuned": False,
            "segment_aggregation_performed": False,
            "gallery_updated": False,
            "identity_assigned": False,
            "team_assigned": False,
            "global_id_modified": False,
            "previous_baseline_modified": False,
            "accuracy_claimed": False,
        },
    }
    jm.write_json(output_dir / RUN_MANIFEST_NAME, run_manifest)

    leftovers = sorted(
        path.name for path in output_dir.iterdir() if path.name not in FINAL_ARTIFACTS
    )
    if leftovers:
        _fail(f"unexpected files in output dir: {leftovers}")
    os.replace(output_dir, final_dir)
    print(f"FINALIZE_OK final={final_dir} status={overall_status}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["prepare", "infer", "evaluate", "finalize"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-manifest", default=None)
    parser.add_argument("--evaluation-reference", default=None)
    parser.add_argument("--baseline-freeze", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline-required", action="store_true")
    parser.add_argument("--strace-log", default=None)
    parser.add_argument("--infer-log", default=None)
    parser.add_argument("--cache-audit", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.baseline_freeze:
        expected_freeze = str(_PROJECT_ROOT / config["baseline_freeze"]["path"])
        if str(Path(args.baseline_freeze).resolve()) != str(Path(expected_freeze).resolve()):
            _fail("--baseline-freeze does not match config baseline_freeze.path")

    if args.phase == "prepare":
        phase_prepare(args, config)
    elif args.phase == "infer":
        if not args.inference_manifest or not args.asset_manifest:
            _fail("--inference-manifest and --asset-manifest are required for infer")
        phase_infer(args, config)
    elif args.phase == "evaluate":
        if not args.evaluation_reference:
            _fail("--evaluation-reference is required for evaluate")
        phase_evaluate(args, config)
    elif args.phase == "finalize":
        if not args.strace_log:
            _fail("--strace-log is required for finalize")
        phase_finalize(args, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
