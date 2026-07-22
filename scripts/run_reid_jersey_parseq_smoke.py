#!/usr/bin/env python
"""Stage 5C-C3D offline SoccerNet-finetuned PARSeq recognizer-only smoke runner.

Phases:
- prepare   : copy byte-identical C1 freeze blind manifest + reference
- preflight : static audit + offline load + synthetic inference
- infer     : 46 blind ROI predictions (model process; wrap with strace)
- evaluate  : join reference after predictions are sealed
- finalize  : network audit, summaries, atomic temp→final rename
- run       : orchestrate child processes with offline env + strace
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid import jersey_parseq as jp

BLIND_MANIFEST_NAME = "parseq_inference_manifest.jsonl"
REFERENCE_NAME = "parseq_evaluation_reference.jsonl"
PREDICTIONS_NAME = "parseq_predictions.jsonl"
ITEM_EVALUATION_NAME = "parseq_item_evaluation.jsonl"
RESULTS_SUMMARY_NAME = "parseq_results_summary.json"
RUNTIME_SUMMARY_NAME = "parseq_runtime_summary.json"
CHECKPOINT_CONTRACT_NAME = "parseq_checkpoint_contract.json"
RUN_MANIFEST_NAME = "parseq_run_manifest.json"
NETWORK_AUDIT_NAME = "parseq_network_audit.txt"
STATIC_AUDIT_NAME = "parseq_static_checkpoint_audit.json"
INFER_META_NAME = "_infer_meta.json"
PREFLIGHT_META_NAME = "_preflight_meta.json"

FINAL_ARTIFACTS = (
    BLIND_MANIFEST_NAME,
    REFERENCE_NAME,
    PREDICTIONS_NAME,
    ITEM_EVALUATION_NAME,
    RESULTS_SUMMARY_NAME,
    RUNTIME_SUMMARY_NAME,
    CHECKPOINT_CONTRACT_NAME,
    RUN_MANIFEST_NAME,
    NETWORK_AUDIT_NAME,
    STATIC_AUDIT_NAME,
)


def _fail(message: str) -> None:
    raise jp.JerseyPARSeqError(message)


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "reid_jersey_parseq_smoke_config_v1":
        _fail("unexpected config schema_version")
    if config.get("device") != "cpu":
        _fail("only device=cpu is allowed")
    if not config.get("offline_required"):
        _fail("offline_required must be true")
    if config["digit_policy"].get("confidence_threshold") is not None:
        _fail("confidence_threshold must be null")
    if config["digit_policy"].get("letter_to_digit_conversion"):
        _fail("letter_to_digit_conversion must be false")
    return config


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = jp.sha256_file(path)
    if actual != expected:
        _fail(f"{label} sha256 mismatch: expected {expected}, got {actual}")


def _expected_counts(config: dict) -> dict[str, int]:
    return {str(k): int(v) for k, v in config["selection"]["expected_counts"].items()}


def phase_prepare(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    freeze = _PROJECT_ROOT / config["baseline_freeze"]["path"]
    blind_src = freeze / "smoke_inference_manifest.jsonl"
    ref_src = freeze / "smoke_evaluation_reference.jsonl"
    _verify_sha(blind_src, config["baseline_freeze"]["inference_manifest_sha256"], "C1 blind")
    _verify_sha(ref_src, config["baseline_freeze"]["evaluation_reference_sha256"], "C1 reference")
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
        _fail("reference copy is not byte-identical")

    blind = jp.load_jsonl(output_dir / BLIND_MANIFEST_NAME)
    refs = jp.load_jsonl(output_dir / REFERENCE_NAME)
    if len(blind) != int(config["max_items"]) or len(refs) != int(config["max_items"]):
        _fail("expected 46 rows in freeze manifests")
    jp.assert_blind_records_safe(blind)
    class_counts: dict[str, int] = {}
    for row in blind:
        class_counts[row["selection_class"]] = class_counts.get(row["selection_class"], 0) + 1
    if class_counts != _expected_counts(config):
        _fail(f"class distribution mismatch: {class_counts}")
    if len({row["review_item_id"] for row in blind}) != 46:
        _fail("duplicate review_item_id in blind manifest")
    print(f"PREPARE_OK items={len(blind)}", flush=True)


def phase_preflight(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    ckpt = Path(config["checkpoint"]["path"])
    _verify_sha(
        Path(config["checkpoint"]["asset_manifest_path"]),
        config["checkpoint"]["asset_manifest_sha256"],
        "checkpoint asset manifest",
    )
    env_man = Path(config["environment"]["manifest_path"])
    env_sha = jp.sha256_file(env_man)
    if not env_sha.startswith(config["environment"]["manifest_sha256_prefix"]):
        _fail(f"environment manifest sha prefix mismatch: {env_sha}")

    generic = Path(config["parseq_root"]) / "models" / "parseq-bb5792a6.pt"
    generic_before = (generic.stat().st_mtime_ns, generic.stat().st_size, jp.sha256_file(generic))

    static_audit = jp.static_checkpoint_audit(ckpt)
    jp.write_json(output_dir / STATIC_AUDIT_NAME, static_audit)

    model, model_meta = jp.load_parseq_recognizer(
        ckpt,
        config["parseq_root"],
        expected_sha256=config["checkpoint"]["sha256"],
        expected_byte_size=int(config["checkpoint"]["byte_size"]),
    )
    synthetic = jp.run_synthetic_inference_preflight(model)
    generic_after = (generic.stat().st_mtime_ns, generic.stat().st_size, jp.sha256_file(generic))
    if generic_before != generic_after:
        _fail("generic parseq-bb5792a6.pt changed during load")

    contract = {
        "schema": jp.CHECKPOINT_CONTRACT_SCHEMA,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "checkpoint_byte_size": int(config["checkpoint"]["byte_size"]),
        "static_archive_audit_status": static_audit["status"],
        "pickle_global_inventory": static_audit["global_refs"],
        "source_trust": {
            "source": static_audit["source"],
            "official_checksum_available": False,
            "official_checksum": None,
            "local_sha256": config["checkpoint"]["sha256"],
            "residual_risk": static_audit["residual_risk"],
        },
        "loader": model_meta["load_from_checkpoint"],
        "model_class": model_meta["model_class"],
        "model_module": model_meta["model_module"],
        "bundled_module_origins": model_meta["bundled_module_origins"],
        "load_warnings": [],
        "missing_keys": model_meta["missing_keys"],
        "unexpected_keys": model_meta["unexpected_keys"],
        "hparams_runtime_contract": model_meta["runtime_contract"],
        "image_transform_contract": {
            "interpolation": "BICUBIC",
            "normalize_mean": 0.5,
            "normalize_std": 0.5,
            "img_size": model_meta["runtime_contract"]["img_size"],
        },
        "cpu_status": True,
        "parameter_count": model_meta["parameter_count"],
        "trainable_parameter_count": model_meta["trainable_parameter_count"],
        "generic_pretrained_fallback_used": False,
        "synthetic_inference": synthetic,
        "load_seconds": model_meta["load_seconds"],
        "contract_status": model_meta["runtime_contract"]["contract_status"],
    }
    jp.write_json(output_dir / CHECKPOINT_CONTRACT_NAME, contract)
    jp.write_json(
        output_dir / PREFLIGHT_META_NAME,
        {"model_meta": model_meta, "synthetic": synthetic, "generic_unchanged": True},
    )
    # Keep model out of this process for infer phase isolation; release reference.
    del model
    print("PREFLIGHT_OK", flush=True)


def phase_infer(args: argparse.Namespace, config: dict) -> None:
    if args.device != "cpu" or not args.offline_required:
        _fail("infer requires --device cpu --offline-required")
    output_dir = Path(args.output_dir)
    blind = jp.load_jsonl(Path(args.inference_manifest))
    jp.assert_blind_records_safe(blind)
    if len(blind) != int(config["max_items"]):
        _fail("blind manifest size mismatch")

    import torch

    if torch.cuda.is_available():
        _fail("CUDA must not be available")
    torch.manual_seed(0)

    # Re-load once for the 46-item batch (single instance reuse).
    model, model_meta = jp.load_parseq_recognizer(
        config["checkpoint"]["path"],
        config["parseq_root"],
        expected_sha256=config["checkpoint"]["sha256"],
        expected_byte_size=int(config["checkpoint"]["byte_size"]),
    )
    start = time.perf_counter()
    predictions = jp.run_blind_inference(blind, model, model_meta)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if len(predictions) != int(config["expected_prediction_count"]):
        _fail(f"expected 46 predictions, got {len(predictions)}")
    jp.write_jsonl(output_dir / PREDICTIONS_NAME, predictions)
    pred_sha = jp.sha256_file(output_dir / PREDICTIONS_NAME)

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
                "pytorch-lightning",
                "timm",
                "numpy",
                "pillow",
            )
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
    }
    jp.write_json(
        output_dir / INFER_META_NAME,
        {
            "model_meta": model_meta,
            "inference_wall_ms": wall_ms,
            "peak_rss_kb": peak_rss_kb,
            "environment": environment,
            "prediction_count": len(predictions),
            "predictions_sha256": pred_sha,
        },
    )
    print(f"INFER_OK predictions={len(predictions)} sha={pred_sha}", flush=True)


def phase_evaluate(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    predictions = jp.load_jsonl(output_dir / PREDICTIONS_NAME)
    # Predictions sealed; only now open the evaluation reference.
    reference_rows = jp.load_jsonl(Path(args.evaluation_reference))
    # Attach source_crop_sha256 onto reference join via blind manifest for audit.
    blind = jp.load_jsonl(output_dir / BLIND_MANIFEST_NAME)
    blind_sha = {row["review_item_id"]: row["source_crop_sha256"] for row in blind}
    enriched = []
    for row in reference_rows:
        copy = dict(row)
        copy["source_crop_sha256"] = blind_sha[row["review_item_id"]]
        enriched.append(copy)
    item_rows, summary = jp.evaluate_predictions(
        predictions, enriched, _expected_counts(config)
    )
    # Fill checkpoint identity / C1 comparison into summary.
    contract = json.loads((output_dir / CHECKPOINT_CONTRACT_NAME).read_text(encoding="utf-8"))
    summary["checkpoint_identity"] = {
        "path": contract["checkpoint_path"],
        "sha256": contract["checkpoint_sha256"],
        "byte_size": contract["checkpoint_byte_size"],
        "model_class": contract["model_class"],
        "contract_status": contract["contract_status"],
    }
    summary["comparison_with_c1_c2_descriptive_only"]["c1_baseline"] = config["c1_baseline_metrics"]
    jp.write_jsonl(output_dir / ITEM_EVALUATION_NAME, item_rows)
    jp.write_json(output_dir / RESULTS_SUMMARY_NAME, summary)
    print(
        f"EVALUATE_OK exact={summary['positive_metrics']['exact_match_count']} "
        f"wrong={summary['positive_metrics']['wrong_number_count']} "
        f"none={summary['positive_metrics']['no_prediction_count']} "
        f"neg_emit={summary['negative_metrics']['accepted_number_emission_count']} "
        f"decision={summary['decision_class']}",
        flush=True,
    )


def phase_finalize(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    final_dir = _PROJECT_ROOT / config["output"]["final_dir"]
    if final_dir.exists():
        _fail(f"final output already exists (no overwrite): {final_dir}")

    strace_text = Path(args.strace_log).read_text(encoding="utf-8", errors="replace")
    audit = jp.parse_network_strace(strace_text)
    if audit["policy_status"] not in {"pass_offline", "pass_loopback_only"}:
        _fail(f"network audit failed: {json.dumps(audit)}")

    infer_meta = json.loads((output_dir / INFER_META_NAME).read_text(encoding="utf-8"))
    summary = json.loads((output_dir / RESULTS_SUMMARY_NAME).read_text(encoding="utf-8"))
    predictions = jp.load_jsonl(output_dir / PREDICTIONS_NAME)

    # Leakage scan
    forbidden = ("manual_jersey_number", "exact_match", "identity", "global_id", "team")
    for row in predictions:
        blob = json.dumps(row)
        for key in forbidden:
            if key in row:
                _fail(f"prediction leakage key {key}")
        if '"manual_jersey_number"' in blob:
            _fail("manual_jersey_number leaked into predictions JSON")

    # NaN/Inf scan on confidences
    for row in predictions:
        conf = row.get("sequence_confidence")
        if conf is not None:
            if conf != conf or conf in (float("inf"), float("-inf")):
                _fail("NaN/Inf confidence in predictions")

    runtime = {
        "schema_version": jp.RUNTIME_SUMMARY_SCHEMA_VERSION,
        "overall_status": summary["status"],
        "checkpoint_load_seconds": infer_meta["model_meta"]["load_seconds"],
        "inference_wall_ms": infer_meta["inference_wall_ms"],
        "per_item_inference_ms": jp.runtime_stats([p.get("inference_ms") for p in predictions]),
        "peak_rss_kb": infer_meta["peak_rss_kb"],
        "environment": infer_meta["environment"],
        "network_policy": audit["policy_status"],
        "unshare_supported": False,
    }
    jp.write_json(output_dir / RUNTIME_SUMMARY_NAME, runtime)

    audit_lines = [f"{key}={json.dumps(value)}" for key, value in audit.items() if key != "violations"]
    audit_lines.append(f"violations_count={len(audit['violations'])}")
    audit_lines.extend(f"violation: {line}" for line in audit["violations"])
    (output_dir / NETWORK_AUDIT_NAME).write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    git_head = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ext_head = subprocess.run(
        ["git", "-C", str(config["external_repo"]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if ext_head != config["external_repo_head"]:
        _fail(f"external HEAD drift: {ext_head}")

    (output_dir / INFER_META_NAME).unlink(missing_ok=True)
    (output_dir / PREFLIGHT_META_NAME).unlink(missing_ok=True)

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
            "path": str(final_dir / name),
            "sha256": jp.sha256_file(path),
            "byte_size": path.stat().st_size,
            "row_count": row_count(path),
        }

    # Exact JSONL row counts
    for name, expected in (
        (BLIND_MANIFEST_NAME, 46),
        (REFERENCE_NAME, 46),
        (PREDICTIONS_NAME, 46),
        (ITEM_EVALUATION_NAME, 46),
    ):
        if artifact_hashes[name]["row_count"] != expected:
            _fail(f"{name} row_count={artifact_hashes[name]['row_count']} expected {expected}")

    run_manifest = {
        "schema_version": jp.RUN_MANIFEST_SCHEMA_VERSION,
        "stage": "5C-C3D",
        "created_at": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(),
        "project_git_head": git_head,
        "external_repo_head": ext_head,
        "environment_name": config["environment"]["name"],
        "environment_manifest_path": config["environment"]["manifest_path"],
        "environment_manifest_sha256": jp.sha256_file(config["environment"]["manifest_path"]),
        "checkpoint_asset_manifest_path": config["checkpoint"]["asset_manifest_path"],
        "checkpoint_asset_manifest_sha256": config["checkpoint"]["asset_manifest_sha256"],
        "input_source": {
            "c1_freeze": str(_PROJECT_ROOT / config["baseline_freeze"]["path"]),
            "inference_manifest_sha256": config["baseline_freeze"]["inference_manifest_sha256"],
            "evaluation_reference_sha256": config["baseline_freeze"]["evaluation_reference_sha256"],
        },
        "artifacts": artifact_hashes,
        "git_status_note": "tracked scope limited to four C3D files; verified by caller",
        "network_policy": audit["policy_status"],
        "network_audit": {k: v for k, v in audit.items() if k != "violations"},
        "temp_cache_note": "temporary HOME/XDG_CACHE_HOME/TORCH_HOME/TMPDIR used in child",
        "source_immutability": {
            "c1_freeze_unmodified": True,
            "checkpoint_unmodified": True,
            "external_repo_unmodified": True,
        },
        "model_initialized": True,
        "checkpoint_loaded": True,
        "inference_performed": True,
        "automatic_download": False,
        "finalization_atomic": True,
        "evidence_labels": summary["evidence_labels"],
        "decision_class": summary["decision_class"],
        "safety_flags": summary["safety_flags"],
    }
    jp.write_json(output_dir / RUN_MANIFEST_NAME, run_manifest)

    leftovers = sorted(p.name for p in output_dir.iterdir() if p.name not in FINAL_ARTIFACTS)
    if leftovers:
        _fail(f"unexpected files in output dir: {leftovers}")
    if len(list(output_dir.iterdir())) != 10:
        _fail("final artifact count must be exactly 10")
    os.replace(output_dir, final_dir)
    print(
        f"FINALIZE_OK final={final_dir} decision={summary['decision_class']} "
        f"policy={audit['policy_status']}",
        flush=True,
    )


def _env_python(config: dict) -> str:
    # Prefer explicit conda env interpreter for PARSeq CPU stack.
    candidate = Path.home() / "miniconda3" / "envs" / config["environment"]["name"] / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _run_child(
    phase: str,
    config_path: Path,
    output_dir: Path,
    *,
    env: dict[str, str],
    python_bin: str,
    strace_log: Path | None = None,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [
        python_bin,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    if strace_log is not None:
        cmd = [
            "strace",
            "-f",
            "-yy",
            "-s",
            "256",
            "-e",
            "trace=network",
            "-o",
            str(strace_log),
            *cmd,
        ]
    result = subprocess.run(cmd, env=env, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        _fail(f"phase {phase} failed with code {result.returncode}")


def phase_run(args: argparse.Namespace, config: dict) -> None:
    final_dir = _PROJECT_ROOT / config["output"]["final_dir"]
    if final_dir.exists():
        _fail(f"final output already exists: {final_dir}")

    unique = f"{int(time.time())}_{os.getpid()}"
    temp_parent = _PROJECT_ROOT / "outputs" / "reid" / "full_stage4b"
    temp_parent.mkdir(parents=True, exist_ok=True)
    output_dir = temp_parent / f"_tmp_jersey_parseq_smoke_stage5c_c3d_{unique}"
    work_home = Path(tempfile.mkdtemp(prefix="c3d_home_"))
    strace_log = work_home / "strace_network.txt"
    python_bin = _env_python(config)
    try:
        child_env = os.environ.copy()
        child_env.update(
            {
                "HOME": str(work_home),
                "XDG_CACHE_HOME": str(work_home / "cache"),
                "TORCH_HOME": str(work_home / "torch"),
                "TMPDIR": str(work_home / "tmp"),
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "http_proxy": "http://127.0.0.1:9",
                "https_proxy": "http://127.0.0.1:9",
                "all_proxy": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
                "PYTHONPATH": f"{config['parseq_root']}:{_SRC_DIR}",
                "CONDA_DEFAULT_ENV": config["environment"]["name"],
            }
        )
        (work_home / "cache").mkdir()
        (work_home / "torch").mkdir()
        (work_home / "tmp").mkdir()

        _run_child("prepare", Path(args.config), output_dir, env=child_env, python_bin=python_bin)
        _run_child(
            "preflight",
            Path(args.config),
            output_dir,
            env=child_env,
            python_bin=python_bin,
            strace_log=strace_log,
        )
        infer_strace = work_home / "strace_infer.txt"
        _run_child(
            "infer",
            Path(args.config),
            output_dir,
            env=child_env,
            python_bin=python_bin,
            strace_log=infer_strace,
            extra_args=[
                "--device",
                "cpu",
                "--offline-required",
                "--inference-manifest",
                str(output_dir / BLIND_MANIFEST_NAME),
            ],
        )
        combined = strace_log.read_text(errors="replace") + infer_strace.read_text(errors="replace")
        strace_log.write_text(combined)

        _run_child(
            "evaluate",
            Path(args.config),
            output_dir,
            env=child_env,
            python_bin=python_bin,
            extra_args=[
                "--evaluation-reference",
                str(output_dir / REFERENCE_NAME),
            ],
        )
        _run_child(
            "finalize",
            Path(args.config),
            output_dir,
            env=child_env,
            python_bin=python_bin,
            extra_args=["--strace-log", str(strace_log)],
        )
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work_home, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("prepare", "preflight", "infer", "evaluate", "finalize", "run"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline-required", action="store_true")
    parser.add_argument("--inference-manifest")
    parser.add_argument("--evaluation-reference")
    parser.add_argument("--strace-log")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _load_config(args.config)
    phases = {
        "prepare": phase_prepare,
        "preflight": phase_preflight,
        "infer": phase_infer,
        "evaluate": phase_evaluate,
        "finalize": phase_finalize,
        "run": phase_run,
    }
    phases[args.phase](args, config)


if __name__ == "__main__":
    main()
