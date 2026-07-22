#!/usr/bin/env python
"""Stage 5C-C1 offline CPU jersey OCR smoke runner (local DBNet + SAR).

The run is split into four phases so that the model process never sees manual
labels and the evaluation reference is only opened after predictions complete:

- ``prepare``  : build the deterministic 46-item selection from the frozen pilot
                 annotations and write the blind inference manifest plus the
                 separate evaluation reference into the temp output dir.
- ``infer``    : the model process. Reads ONLY the blind inference manifest and
                 the config; initializes DBNet/SAR from validated local paths;
                 writes ``smoke_predictions.jsonl`` and internal runtime metadata.
                 Wrap this phase with strace for the network audit.
- ``evaluate`` : joins predictions with the evaluation reference and writes
                 ``smoke_item_evaluation.jsonl`` and ``smoke_results_summary.json``.
- ``finalize`` : classifies the strace log under the loopback-only policy,
                 writes ``smoke_network_audit.txt``, ``smoke_runtime_summary.json``
                 and ``smoke_run_manifest.json``, validates the output contract
                 and atomically renames the temp dir to the final output dir.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
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

BLIND_MANIFEST_NAME = "smoke_inference_manifest.jsonl"
REFERENCE_NAME = "smoke_evaluation_reference.jsonl"
PREDICTIONS_NAME = "smoke_predictions.jsonl"
ITEM_EVALUATION_NAME = "smoke_item_evaluation.jsonl"
RESULTS_SUMMARY_NAME = "smoke_results_summary.json"
RUNTIME_SUMMARY_NAME = "smoke_runtime_summary.json"
RUN_MANIFEST_NAME = "smoke_run_manifest.json"
NETWORK_AUDIT_NAME = "smoke_network_audit.txt"
INFER_META_NAME = "_infer_meta.json"

FINAL_ARTIFACTS = (
    BLIND_MANIFEST_NAME,
    REFERENCE_NAME,
    PREDICTIONS_NAME,
    ITEM_EVALUATION_NAME,
    RESULTS_SUMMARY_NAME,
    RUNTIME_SUMMARY_NAME,
    RUN_MANIFEST_NAME,
    NETWORK_AUDIT_NAME,
)

WARNING_PATTERNS = {
    "state_dict_unexpected_key_data_preprocessor": re.compile(
        r"unexpected key in source state_dict:.*data_preprocessor\.mean, data_preprocessor\.std"
    ),
    "mmengine_registry_scope_warning": re.compile(
        r'Failed to search registry with scope "mmocr"'
    ),
    "local_vis_backend_warning": re.compile(r"LocalVisBackend"),
    "state_dict_mismatch_notice": re.compile(
        r"The model and loaded state dict do not match exactly"
    ),
}


def _fail(message: str) -> None:
    raise jm.JerseyMMOCRError(message)


def _expected_counts(config: dict) -> dict[str, int]:
    return {str(k): int(v) for k, v in config["selection"]["expected_counts"].items()}


def _verify_input_sha(path: Path, expected_sha: str, label: str) -> None:
    actual = jm.sha256_file(path)
    if actual != expected_sha:
        _fail(f"{label} sha256 mismatch: expected {expected_sha}, got {actual}")


def phase_prepare(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    reviewed_path = _PROJECT_ROOT / config["inputs"]["pilot_reviewed_items"]
    canonical_path = _PROJECT_ROOT / config["inputs"]["canonical_review_items"]
    _verify_input_sha(reviewed_path, config["inputs"]["pilot_reviewed_items_sha256"], "pilot_reviewed_items")
    _verify_input_sha(canonical_path, config["inputs"]["canonical_review_items_sha256"], "canonical_review_items")

    reviewed_rows = jm.load_jsonl(reviewed_path)
    canonical_by_id = {row["review_item_id"]: row for row in jm.load_jsonl(canonical_path)}

    expected_counts = _expected_counts(config)
    selected = jm.build_selection(
        reviewed_rows,
        expected_counts,
        a_max_per_segment=int(config["selection"]["a_not_visible_max_per_segment"]),
    )
    blind_records = jm.build_blind_manifest(selected, canonical_by_id)
    reference_records = jm.build_evaluation_reference(selected)
    jm.validate_manifest_pair(blind_records, reference_records, int(config["max_items"]))

    jm.write_jsonl(output_dir / BLIND_MANIFEST_NAME, blind_records)
    jm.write_jsonl(output_dir / REFERENCE_NAME, reference_records)
    print(f"PREPARE_OK items={len(blind_records)}", flush=True)


def phase_infer(args: argparse.Namespace, config: dict) -> None:
    if args.device != "cpu":
        _fail("only --device cpu is allowed")
    if not args.offline_required:
        _fail("--offline-required must be set for the smoke run")
    output_dir = Path(args.output_dir)

    asset_manifest_path = Path(args.asset_manifest)
    _verify_input_sha(asset_manifest_path, config["asset_manifest_sha256"], "asset_manifest")
    _verify_input_sha(
        Path(config["config_closure_path"]), config["config_closure_sha256"], "config_closure"
    )

    # The model process reads ONLY the blind manifest (plus config); the
    # evaluation reference is never opened in this phase.
    blind_records = jm.load_jsonl(Path(args.inference_manifest))
    jm.assert_blind_records_safe(blind_records)
    if len(blind_records) != int(args.max_items):
        _fail(f"blind manifest must have {args.max_items} rows, got {len(blind_records)}")

    import torch

    if torch.cuda.is_available():
        _fail("CUDA must not be available in the smoke environment")
    torch.manual_seed(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector, recognizer, init_timings = jm.build_inferencers(config)
    python_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    model_meta = jm.build_model_meta(config)
    start = time.perf_counter()
    predictions = jm.run_blind_inference(blind_records, detector, recognizer, model_meta)
    wall_ms = (time.perf_counter() - start) * 1000.0
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
            for name in ("torch", "torchvision", "numpy", "opencv-python", "mmcv", "mmengine", "mmdet", "mmocr")
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
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
    print(f"INFER_OK items={len(predictions)}", flush=True)


def phase_evaluate(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    predictions = jm.load_jsonl(output_dir / PREDICTIONS_NAME)
    # Predictions are complete on disk; only now is the reference opened.
    reference_rows = jm.load_jsonl(Path(args.evaluation_reference))
    item_rows, results_summary = jm.evaluate_predictions(
        predictions, reference_rows, _expected_counts(config)
    )
    jm.write_jsonl(output_dir / ITEM_EVALUATION_NAME, item_rows)
    jm.write_json(output_dir / RESULTS_SUMMARY_NAME, results_summary)
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

    infer_meta_path = output_dir / INFER_META_NAME
    infer_meta = json.loads(infer_meta_path.read_text(encoding="utf-8"))
    predictions = jm.load_jsonl(output_dir / PREDICTIONS_NAME)

    warning_summary = {
        "python_warnings": infer_meta["python_warnings"],
        "log_warning_counts": log_warning_counts,
        "state_dict_unexpected_keys": {
            "message": "unexpected key in source state_dict: data_preprocessor.mean, data_preprocessor.std",
            "occurrence_count": log_warning_counts["state_dict_unexpected_key_data_preprocessor"],
            "affected_checkpoints": ["dbnet", "sar"]
            if log_warning_counts["state_dict_unexpected_key_data_preprocessor"] >= 2
            else (["one_of_dbnet_sar"] if log_warning_counts["state_dict_unexpected_key_data_preprocessor"] == 1 else []),
            "fail": False,
            "note": "model init completed; data_preprocessor mean/std provided by config",
        },
    }
    runtime_summary = jm.build_runtime_summary(
        predictions,
        infer_meta["init_timings"],
        infer_meta["environment"],
        infer_meta["peak_rss_kb"],
        warning_summary,
    )
    jm.write_json(output_dir / RUNTIME_SUMMARY_NAME, runtime_summary)

    audit_lines = [f"{key}={json.dumps(value)}" for key, value in audit.items() if key != "violations"]
    audit_lines.append(f"violations_count={len(audit['violations'])}")
    audit_lines.extend(f"violation: {line}" for line in audit["violations"])
    (output_dir / NETWORK_AUDIT_NAME).write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    git_head = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    infer_meta_path.unlink()

    artifact_hashes = {}
    for name in FINAL_ARTIFACTS:
        path = output_dir / name
        if name in (RUN_MANIFEST_NAME,):
            continue
        if not path.is_file():
            _fail(f"missing artifact before finalize: {name}")
        artifact_hashes[name] = {
            "sha256": jm.sha256_file(path),
            "byte_size": path.stat().st_size,
        }

    run_manifest = {
        "schema_version": jm.RUN_MANIFEST_SCHEMA_VERSION,
        "stage": "5C-C1",
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "project_git_head": git_head,
        "config_path": str(Path(args.config)),
        "config_sha256": jm.sha256_file(args.config),
        "environment": infer_meta["environment"],
        "models": jm.build_model_meta(config),
        "asset_manifest_sha256": config["asset_manifest_sha256"],
        "config_closure_sha256": config["config_closure_sha256"],
        "inputs": config["inputs"],
        "selection_expected_counts": _expected_counts(config),
        "network_isolation": {
            "unshare_supported": False,
            "unshare_note": "unshare -n returned 'Operation not permitted' on this system; not used",
            "methods": [
                "temporary HOME/XDG_CACHE_HOME/TORCH_HOME/TMPDIR",
                "invalid HTTP/HTTPS/ALL proxy",
                "strace -f -yy -s 256 -e trace=network",
            ],
            "policy": "loopback_only",
            "audit": {key: value for key, value in audit.items() if key != "violations"},
            "audit_violation_count": len(audit["violations"]),
        },
        "warnings": warning_summary,
        "artifacts": artifact_hashes,
        "safety_flags": {
            "manual_label_blind_inference": True,
            "image_export": False,
            "source_overwrite": False,
            "threshold_tuning": False,
            "identity_gallery_global_id_changes": False,
            "commit_push_performed": False,
        },
    }
    jm.write_json(output_dir / RUN_MANIFEST_NAME, run_manifest)

    leftovers = sorted(
        path.name for path in output_dir.iterdir() if path.name not in FINAL_ARTIFACTS
    )
    if leftovers:
        _fail(f"unexpected files in output dir: {leftovers}")
    os.replace(output_dir, final_dir)
    print(f"FINALIZE_OK final={final_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["prepare", "infer", "evaluate", "finalize"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, help="temp output dir (atomic rename target source)")
    parser.add_argument("--inference-manifest", default=None)
    parser.add_argument("--evaluation-reference", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-items", type=int, default=46)
    parser.add_argument("--offline-required", action="store_true")
    parser.add_argument("--strace-log", default=None)
    parser.add_argument("--infer-log", default=None)
    args = parser.parse_args()

    config = jm.load_smoke_config(args.config)
    if int(config["max_items"]) != int(args.max_items):
        _fail(f"config max_items {config['max_items']} != --max-items {args.max_items}")

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
