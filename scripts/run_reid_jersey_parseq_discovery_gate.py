#!/usr/bin/env python
"""Stage 5C-R6 discovery-primary PARSeq inference + candidate-gate derivation.

Discovery-only (40 items). No reserve/holdout access. No C3D/C3E threshold reuse.
Uses generic PARSeq adapter utilities from football_analytics.reid.jersey_parseq.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid import jersey_parseq as jp  # noqa: E402

DIGIT_RE = re.compile(r"^[0-9]{1,2}$")
CONFIG_SCHEMA = "reid_jersey_parseq_discovery_gate_config_v1"
CONTRACT_SCHEMA = "reid_jersey_parseq_discovery_contract_v1"
SUMMARY_SCHEMA = "reid_jersey_parseq_discovery_summary_v1"
MANIFEST_SCHEMA = "reid_jersey_parseq_discovery_manifest_v1"
GATE_SCHEMA = "reid_jersey_parseq_discovery_candidate_gate_v1"


class DiscoveryGateError(RuntimeError):
    """Contract / integrity failure for discovery PARSeq gate."""


def sha256_file(path: Path | str) -> str:
    return jp.sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def float64_hex(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def exact_decimal(value: float) -> str:
    return format(float(value), ".17g")


def listing_sha(root: Path) -> tuple[int, str]:
    files: list[tuple[str, int, str]] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = Path(dp) / fn
            rel = str(p.relative_to(root))
            files.append((rel, p.stat().st_size, sha256_file(p)))
    files.sort()
    blob = "\n".join(f"{a}\t{b}\t{c}" for a, b, c in files).encode()
    return len(files), hashlib.sha256(blob).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise DiscoveryGateError("unexpected config schema")
    if config.get("device") != "cpu" or not config.get("offline_required"):
        raise DiscoveryGateError("cpu/offline required")
    return config


def annotation_class(readable: str) -> str:
    if readable == "yes":
        return "readable_positive"
    if readable == "no":
        return "non_readable_negative"
    if readable == "uncertain":
        return "uncertain_excluded"
    raise DiscoveryGateError(f"bad readable={readable!r}")


def build_input_universe(
    *,
    project_root: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canon = project_root / config["canonical_split"]["path"]
    ann_root = project_root / config["annotation_freeze"]["path"]
    man_path = canon / config["canonical_split"]["discovery_primary_manifest"]
    frozen_path = ann_root / config["annotation_freeze"]["frozen_csv"]
    universe_path = project_root / config["clean_universe_items"]

    manifest = [
        json.loads(line)
        for line in man_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with frozen_path.open(encoding="utf-8", newline="") as handle:
        frozen = list(csv.DictReader(handle))
    universe = {
        json.loads(line)["review_item_id"]: json.loads(line)
        for line in universe_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(manifest) != 40 or len(frozen) != 40:
        raise DiscoveryGateError("BLOCKED_DISCOVERY_INPUT_CONTRACT_MISMATCH size")
    man_ids = [r["split_item_id"] for r in manifest]
    fr_ids = [r["split_item_id"] for r in frozen]
    if len(set(man_ids)) != 40 or len(set(fr_ids)) != 40:
        raise DiscoveryGateError("BLOCKED_DISCOVERY_INPUT_CONTRACT_MISMATCH duplicate")
    if set(man_ids) != set(fr_ids):
        raise DiscoveryGateError("BLOCKED_DISCOVERY_INPUT_CONTRACT_MISMATCH id set")
    frozen_by = {r["split_item_id"]: r for r in frozen}
    rows: list[dict[str, Any]] = []
    for man in manifest:
        sid = man["split_item_id"]
        fr = frozen_by[sid]
        if str(man["review_item_id"]) != str(fr["review_item_id"]):
            raise DiscoveryGateError(f"review_item mismatch {sid}")
        if str(man["source_crop_path"]) != str(fr["source_crop_path"]):
            raise DiscoveryGateError(f"path mismatch {sid}")
        if str(man["source_crop_sha256"]) != str(fr["source_crop_sha256"]):
            raise DiscoveryGateError(f"sha mismatch {sid}")
        u = universe[str(man["review_item_id"])]
        if not bool(u.get("roi_valid")):
            raise DiscoveryGateError(f"roi_valid false {sid}")
        crop = Path(str(man["source_crop_path"]))
        if not crop.is_file() or crop.is_symlink():
            raise DiscoveryGateError(f"crop missing {sid}")
        if sha256_file(crop) != str(man["source_crop_sha256"]):
            raise DiscoveryGateError(f"crop sha fail {sid}")
        cls = annotation_class(str(fr["manual_number_readable"]))
        rows.append(
            {
                "split_item_id": sid,
                "review_item_id": str(man["review_item_id"]),
                "crop_id": str(man["crop_id"]),
                "segment_id": str(man["segment_id"]),
                "raw_track_id": int(man["raw_track_id"]),
                "frame_index": int(man["frame_index"]),
                "source_crop_path": str(man["source_crop_path"]),
                "source_crop_sha256": str(man["source_crop_sha256"]),
                "source_crop_width": int(u["source_crop_width"]),
                "source_crop_height": int(u["source_crop_height"]),
                "roi_x_min": int(u["number_roi_x"]),
                "roi_y_min": int(u["number_roi_y"]),
                "roi_x_max": int(u["number_roi_x_max"]),
                "roi_y_max": int(u["number_roi_y_max"]),
                "roi_valid": True,
                "manual_crop_valid": fr["manual_crop_valid"],
                "manual_number_visible": fr["manual_number_visible"],
                "manual_number_readable": fr["manual_number_readable"],
                "manual_jersey_number": fr.get("manual_jersey_number") or "",
                "annotation_class": cls,
                "batch_order": int(man["batch_order"]),
            }
        )
    class_counts = Counter(r["annotation_class"] for r in rows)
    if (
        class_counts["readable_positive"] != 10
        or class_counts["non_readable_negative"] != 27
        or class_counts["uncertain_excluded"] != 3
    ):
        raise DiscoveryGateError(f"class counts {dict(class_counts)}")
    return rows


def freeze_preprocessing_contract(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    adapter = project_root / "src/football_analytics/reid/jersey_parseq.py"
    smoke_cfg = project_root / "configs/reid/jersey_parseq_smoke_stage5c_c3d.yaml"
    smoke_runner = project_root / "scripts/run_reid_jersey_parseq_smoke.py"
    pp = dict(config["preprocessing_contract"])
    # Single active tracked contract: C3D smoke adapter.
    contract = {
        "schema_version": "reid_jersey_parseq_preprocessing_contract_pre_inference_v1",
        "predictions_seen_at_freeze": False,
        "frozen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_paths": {
            "adapter": str(adapter),
            "adapter_sha256": sha256_file(adapter),
            "smoke_config": str(smoke_cfg),
            "smoke_config_sha256": sha256_file(smoke_cfg),
            "smoke_runner": str(smoke_runner),
            "smoke_runner_sha256": sha256_file(smoke_runner),
            "discovery_config_sha256": None,  # filled by caller if needed
        },
        "selected_preprocessing_contract": pp,
        "exact_confidence_metric": pp["confidence_metric"],
        "acceptance_prediction_format": {
            "normalize": "NFKC_strip_surrounding_whitespace_only",
            "valid_jersey_regex": config["digit_policy"]["pattern"],
            "leading_zero_preserved": True,
            "letter_to_digit_conversion": False,
        },
        "normalization_rules": {
            "whitespace_strip_only": True,
            "no_digit_character_rewrite": True,
            "09_differs_from_9": True,
        },
        "ambiguous_contracts_found": False,
        "selection_basis": (
            "single_tracked_active_PARSeq_adapter_contract_from_C3D_smoke_reused_for_r2"
        ),
    }
    write_json(out_path, contract)
    return contract


def predict_discovery_item(
    item: Mapping[str, Any],
    model: Any,
    model_meta: Mapping[str, Any],
    transform: Any,
    *,
    preprocessing_contract_sha: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "split_item_id": item["split_item_id"],
        "review_item_id": item["review_item_id"],
        "crop_id": item["crop_id"],
        "segment_id": item["segment_id"],
        "raw_track_id": item["raw_track_id"],
        "frame_index": item["frame_index"],
        "source_crop_path": item["source_crop_path"],
        "source_crop_sha256": item["source_crop_sha256"],
        "source_crop_width": item["source_crop_width"],
        "source_crop_height": item["source_crop_height"],
        "roi_xyxy": [
            item["roi_x_min"],
            item["roi_y_min"],
            item["roi_x_max"],
            item["roi_y_max"],
        ],
        "extracted_roi_pixel_sha256": None,
        "raw_decoded_text": None,
        "normalized_prediction": None,
        "prediction_emitted": False,
        "valid_jersey_string": False,
        "confidence": None,
        "confidence_metric_name": model_meta.get(
            "confidence_metric_name",
            "product_of_tokenizer_decode_selected_token_probabilities",
        ),
        "token_probabilities": None,
        "inference_ms": None,
        "checkpoint_sha256": model_meta["checkpoint_sha256"],
        "preprocessing_contract_sha256": preprocessing_contract_sha,
        "inference_error": None,
    }
    try:
        crop_path = Path(item["source_crop_path"])
        if sha256_file(crop_path) != item["source_crop_sha256"]:
            raise DiscoveryGateError("crop sha mismatch during infer")
        with Image.open(crop_path) as handle:
            rgb_full = np.asarray(handle.convert("RGB"))
        image_bgr = rgb_full[:, :, ::-1].copy()
        h, w = image_bgr.shape[:2]
        if w != int(item["source_crop_width"]) or h != int(item["source_crop_height"]):
            raise DiscoveryGateError(
                f"dimension mismatch {item['split_item_id']}: {w}x{h}"
            )
        roi = jp.clamp_roi(
            int(item["roi_x_min"]),
            int(item["roi_y_min"]),
            int(item["roi_x_max"]),
            int(item["roi_y_max"]),
            w,
            h,
        )
        if roi is None:
            raise DiscoveryGateError(f"roi invalid {item['split_item_id']}")
        x1, y1, x2, y2 = roi
        roi_bgr = image_bgr[y1:y2, x1:x2]
        out["extracted_roi_pixel_sha256"] = sha256_bytes(roi_bgr.tobytes())
        pil_rgb = jp.bgr_to_rgb_pil(roi_bgr)
        tensor = transform(pil_rgb).unsqueeze(0)
        if list(tensor.shape) != [1, 3, 32, 128]:
            raise DiscoveryGateError(f"bad transform shape {list(tensor.shape)}")
        with torch.inference_mode():
            logits = model(tensor)
        decoded = jp.decode_jersey_logits(model, logits)
        raw = decoded["raw_decoded_text"]
        normalized = jp.normalize_recognized_text(raw)
        conf = decoded["sequence_confidence"]
        if conf is not None and not math.isfinite(float(conf)):
            raise DiscoveryGateError("BLOCKED_DISCOVERY_INFERENCE_INTEGRITY nonfinite conf")
        accepted, _rej = jp.extract_digit_candidate(normalized)
        valid = accepted is not None and DIGIT_RE.fullmatch(accepted) is not None
        out.update(
            {
                "raw_decoded_text": raw,
                "normalized_prediction": normalized,
                "prediction_emitted": bool(normalized),
                "valid_jersey_string": bool(valid),
                "accepted_prediction": accepted if valid else None,
                "confidence": None if conf is None else float(conf),
                "token_probabilities": decoded["token_probabilities"],
                "decode_method": decoded["decode_method"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        out["inference_error"] = f"{type(exc).__name__}: {exc}"
    out["inference_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


def run_inference_pass(
    items: Sequence[Mapping[str, Any]],
    model: Any,
    model_meta: Mapping[str, Any],
    *,
    preprocessing_contract_sha: str,
) -> list[dict[str, Any]]:
    transform = jp.get_transform_for_model(model)
    return [
        predict_discovery_item(
            item,
            model,
            model_meta,
            transform,
            preprocessing_contract_sha=preprocessing_contract_sha,
        )
        for item in items
    ]


def assert_deterministic(
    pass_a: Sequence[Mapping[str, Any]],
    pass_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(pass_a) != 40 or len(pass_b) != 40:
        raise DiscoveryGateError("BLOCKED_DISCOVERY_INFERENCE_NONDETERMINISTIC size")
    diffs = []
    for a, b in zip(pass_a, pass_b):
        if a["split_item_id"] != b["split_item_id"]:
            diffs.append({"id": a["split_item_id"], "error": "id_order"})
            continue
        if a.get("normalized_prediction") != b.get("normalized_prediction"):
            diffs.append({"id": a["split_item_id"], "error": "normalized_prediction"})
        if bool(a.get("valid_jersey_string")) != bool(b.get("valid_jersey_string")):
            diffs.append({"id": a["split_item_id"], "error": "valid_flag"})
        ca, cb = a.get("confidence"), b.get("confidence")
        if ca is None and cb is None:
            continue
        if ca is None or cb is None or abs(float(ca) - float(cb)) > 1e-12:
            diffs.append(
                {
                    "id": a["split_item_id"],
                    "error": "confidence",
                    "a": ca,
                    "b": cb,
                }
            )
    if diffs:
        raise DiscoveryGateError(
            "BLOCKED_DISCOVERY_INFERENCE_NONDETERMINISTIC " + json.dumps(diffs[:5])
        )
    return {
        "normalized_prediction_exact_match": "40/40",
        "valid_flag_exact_match": "40/40",
        "confidence_abs_diff_le_1e12": "40/40",
    }


def evaluate_items(
    items: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pred_by = {p["split_item_id"]: p for p in predictions}
    rows = []
    for item in items:
        p = pred_by[item["split_item_id"]]
        cls = item["annotation_class"]
        valid = bool(p.get("valid_jersey_string"))
        pred = p.get("accepted_prediction") if valid else None
        conf = p.get("confidence")
        if conf is not None and not math.isfinite(float(conf)):
            raise DiscoveryGateError("BLOCKED_DISCOVERY_INFERENCE_INTEGRITY conf")
        if cls == "readable_positive":
            label = str(item["manual_jersey_number"])
            if valid and pred == label:
                outcome = "exact"
            elif valid:
                outcome = "wrong"
            else:
                outcome = "no_prediction"
        elif cls == "non_readable_negative":
            outcome = (
                "negative_digit_emission" if valid else "negative_no_prediction"
            )
        else:
            outcome = (
                "uncertain_digit_emission" if valid else "uncertain_no_prediction"
            )
        rows.append(
            {
                "split_item_id": item["split_item_id"],
                "annotation_class": cls,
                "manual_exact_label": (
                    item["manual_jersey_number"]
                    if cls == "readable_positive"
                    else None
                ),
                "prediction": pred,
                "normalized_prediction": p.get("normalized_prediction"),
                "valid_jersey_string": valid,
                "confidence": conf,
                "outcome": outcome,
                "inference_ms": p.get("inference_ms"),
                "extracted_roi_pixel_sha256": p.get("extracted_roi_pixel_sha256"),
            }
        )
    return rows


def auroc(scores: list[float], labels: list[int]) -> Optional[float]:
    """Mann-Whitney AUROC; None if undefined."""
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    if len(set(labels)) < 2:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def build_operating_points(
    eval_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # Valid emission confidences only (finite).
    emission_confs = []
    for row in eval_rows:
        if row["valid_jersey_string"]:
            conf = row["confidence"]
            if conf is None or not math.isfinite(float(conf)):
                raise DiscoveryGateError("BLOCKED_DISCOVERY_INFERENCE_INTEGRITY")
            emission_confs.append(float(conf))
    unique_cuts = sorted(set(emission_confs))
    points: list[dict[str, Any]] = []
    for idx, cut in enumerate(unique_cuts):
        accepted = []
        counts = Counter()
        for row in eval_rows:
            if not row["valid_jersey_string"]:
                continue
            conf = float(row["confidence"])
            if conf >= cut:
                accepted.append(row)
                cls = row["annotation_class"]
                outcome = row["outcome"]
                if cls == "readable_positive" and outcome == "exact":
                    counts["accepted_exact"] += 1
                elif cls == "readable_positive" and outcome == "wrong":
                    counts["accepted_wrong_positive"] += 1
                elif cls == "non_readable_negative":
                    counts["accepted_negative"] += 1
                elif cls == "uncertain_excluded":
                    counts["accepted_uncertain"] += 1
        rejected_exact = sum(
            1
            for row in eval_rows
            if row["outcome"] == "exact"
            and (
                not row["valid_jersey_string"]
                or float(row["confidence"]) < cut
            )
        )
        safety_ok = (
            counts["accepted_wrong_positive"] == 0
            and counts["accepted_negative"] == 0
            and counts["accepted_exact"] > 0
        )
        op_id = f"op_{idx:04d}_cut_{exact_decimal(cut)}"
        points.append(
            {
                "operating_point_id": op_id,
                "confidence_cut": cut,
                "confidence_cut_exact_decimal": exact_decimal(cut),
                "confidence_cut_float64_hex": float64_hex(cut),
                "sentinel_type": None,
                "accepted_total": len(accepted),
                "accepted_exact": counts["accepted_exact"],
                "accepted_wrong_positive": counts["accepted_wrong_positive"],
                "accepted_negative": counts["accepted_negative"],
                "accepted_uncertain": counts["accepted_uncertain"],
                "rejected_exact": rejected_exact,
                "accepted_item_ids": [r["split_item_id"] for r in accepted],
                "safety_eligible": safety_ok,
                "discovery_support": counts["accepted_exact"],
            }
        )
    # Diagnostic no-acceptance sentinel
    if unique_cuts:
        sentinel_cut = max(unique_cuts) + 1.0
    else:
        sentinel_cut = 1.0
    points.append(
        {
            "operating_point_id": "op_sentinel_no_acceptance",
            "confidence_cut": sentinel_cut,
            "confidence_cut_exact_decimal": exact_decimal(sentinel_cut),
            "confidence_cut_float64_hex": float64_hex(sentinel_cut),
            "sentinel_type": "no_acceptance",
            "accepted_total": 0,
            "accepted_exact": 0,
            "accepted_wrong_positive": 0,
            "accepted_negative": 0,
            "accepted_uncertain": 0,
            "rejected_exact": sum(1 for r in eval_rows if r["outcome"] == "exact"),
            "accepted_item_ids": [],
            "safety_eligible": False,
            "discovery_support": 0,
        }
    )
    return points


def derive_candidate_gate(
    operating_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [
        p
        for p in operating_points
        if p.get("sentinel_type") is None
        and int(p["accepted_wrong_positive"]) == 0
        and int(p["accepted_negative"]) == 0
        and int(p["accepted_exact"]) > 0
    ]
    if not eligible:
        # nearest diagnostic: lowest wrong+neg among non-sentinel with exact>0 else any
        candidates = [p for p in operating_points if p.get("sentinel_type") is None]
        nearest = sorted(
            candidates,
            key=lambda p: (
                int(p["accepted_wrong_positive"]) + int(p["accepted_negative"]),
                -int(p["accepted_exact"]),
                float(p["confidence_cut"]),
                str(p["operating_point_id"]),
            ),
        )[:5]
        return {
            "gate_status": "NO_SAFE_DISCOVERY_CANDIDATE_GATE",
            "threshold_selected": False,
            "deployment_threshold_selected": False,
            "holdout_should_not_open": True,
            "discovery_reserve_should_not_open": True,
            "reason": "no_zero_error_operating_point_with_accepted_exact_gt_0",
            "nearest_diagnostic_operating_points": nearest,
            "eligible_operating_point_ids": [],
            "selected_operating_point": None,
        }

    def sort_key(p: Mapping[str, Any]) -> tuple:
        return (
            float(p["confidence_cut"]),
            -int(p["accepted_exact"]),
            float(p["confidence_cut"]),
            str(p["operating_point_id"]),
        )

    chosen = sorted(eligible, key=sort_key)[0]
    return {
        "gate_status": "DISCOVERY_SAFE_CANDIDATE_GATE_DERIVED",
        "threshold_selected": True,
        "deployment_threshold_selected": False,
        "threshold_scope": "holdout_validation_candidate_only",
        "threshold_source": "discovery_primary_r2",
        "candidate_confidence_cut": float(chosen["confidence_cut"]),
        "candidate_confidence_cut_exact_decimal": chosen["confidence_cut_exact_decimal"],
        "candidate_confidence_cut_float64_hex": chosen["confidence_cut_float64_hex"],
        "comparison_operator": ">=",
        "accepted_exact": int(chosen["accepted_exact"]),
        "accepted_wrong_positive": int(chosen["accepted_wrong_positive"]),
        "accepted_negative": int(chosen["accepted_negative"]),
        "accepted_uncertain": int(chosen["accepted_uncertain"]),
        "accepted_item_ids": list(chosen["accepted_item_ids"]),
        "support_count": int(chosen["accepted_exact"]),
        "calibrated_probability": False,
        "deployment_ready": False,
        "holdout_validated": False,
        "mutable_after_holdout": False,
        "eligible_operating_point_ids": [p["operating_point_id"] for p in eligible],
        "selected_operating_point": chosen,
        "selection_rule": [
            "lowest_numeric_confidence_cut",
            "higher_accepted_exact",
            "confidence_cut_ascending",
            "operating_point_id_ascending",
        ],
    }


def confidence_stats(values: Sequence[Optional[float]]) -> Optional[dict[str, float]]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return None
    return {
        "min": min(nums),
        "median": float(statistics.median(nums)),
        "max": max(nums),
        "count": len(nums),
    }


def run_pipeline(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise DiscoveryGateError("FAILED_ATOMIC_DISCOVERY_PARSEQ_GATE final exists")

    # Input integrity
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != config["project_head_expected"]:
        raise DiscoveryGateError("BLOCKED_DISCOVERY_INPUT_CONTRACT_MISMATCH HEAD")
    canon = project_root / config["canonical_split"]["path"]
    n, listing = listing_sha(canon)
    if listing != config["canonical_split"]["listing_sha256"]:
        raise DiscoveryGateError("BLOCKED_DISCOVERY_INPUT_CONTRACT_MISMATCH listing")
    ann_root = project_root / config["annotation_freeze"]["path"]
    ann_n, ann_listing = listing_sha(ann_root)

    # Checkpoint / repo
    repo = Path(config["external_repo"])
    repo_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if repo_head != config["external_repo_head"]:
        raise DiscoveryGateError("BLOCKED_PARSEQ_RUNTIME_OR_CHECKPOINT repo_head")
    repo_porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if repo_porcelain:
        raise DiscoveryGateError("BLOCKED_PARSEQ_RUNTIME_OR_CHECKPOINT repo_dirty")

    ckpt = Path(config["checkpoint"]["path"])
    try:
        jp.validate_checkpoint_asset(
            ckpt,
            config["checkpoint"]["sha256"],
            int(config["checkpoint"]["byte_size"]),
        )
    except jp.JerseyPARSeqError as exc:
        raise DiscoveryGateError(f"BLOCKED_PARSEQ_RUNTIME_OR_CHECKPOINT {exc}") from exc

    env_man = Path(config["environment"]["manifest_path"])
    env_sha = sha256_file(env_man)
    if not env_sha.startswith(config["environment"]["manifest_sha256_prefix"]):
        raise DiscoveryGateError("BLOCKED_PARSEQ_RUNTIME_OR_CHECKPOINT env_manifest")

    items = build_input_universe(project_root=project_root, config=config)

    token = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    tmp = final_dir.parent / (
        f"_tmp_full_stage4b_rebuild_r2_stage5c_discovery_primary_parseq_gate_{token}"
    )
    if tmp.exists():
        raise DiscoveryGateError("temp exists")
    tmp.mkdir(parents=False)
    for sub in (
        "inference",
        "evaluation",
        "gate_derivation",
        "runtime",
        "effective_configs",
    ):
        (tmp / sub).mkdir()

    try:
        shutil.copy2(config_path, tmp / "effective_configs" / config_path.name)
        pp_path = tmp / "runtime" / "preprocessing_contract_pre_inference.json"
        pp_contract = freeze_preprocessing_contract(
            config=config, project_root=project_root, out_path=pp_path
        )
        pp_contract["source_paths"]["discovery_config_sha256"] = sha256_file(config_path)
        write_json(pp_path, pp_contract)
        pp_sha = sha256_file(pp_path)

        write_jsonl(tmp / "runtime" / "discovery_inference_inputs.jsonl", items)

        import torch

        if torch.cuda.is_available():
            raise DiscoveryGateError("BLOCKED_PARSEQ_RUNTIME_OR_CHECKPOINT cuda")
        torch.manual_seed(0)

        model, model_meta = jp.load_parseq_recognizer(
            ckpt,
            config["parseq_root"],
            expected_sha256=config["checkpoint"]["sha256"],
            expected_byte_size=int(config["checkpoint"]["byte_size"]),
        )
        model_meta = dict(model_meta)
        model_meta["confidence_metric_name"] = config["preprocessing_contract"][
            "confidence_metric"
        ]
        syn = jp.run_synthetic_inference_preflight(model)
        write_json(
            tmp / "runtime" / "model_load_meta.json",
            {"model_meta": model_meta, "synthetic": syn},
        )

        preds1 = run_inference_pass(
            items, model, model_meta, preprocessing_contract_sha=pp_sha
        )
        preds2 = run_inference_pass(
            items, model, model_meta, preprocessing_contract_sha=pp_sha
        )
        det = assert_deterministic(preds1, preds2)
        write_json(tmp / "runtime" / "determinism_audit.json", det)
        write_jsonl(tmp / "inference" / "discovery_primary_predictions.jsonl", preds1)
        if any(p.get("inference_error") for p in preds1):
            raise DiscoveryGateError(
                "BLOCKED_DISCOVERY_INFERENCE_INTEGRITY "
                + str([p["split_item_id"] for p in preds1 if p.get("inference_error")])
            )

        eval_rows = evaluate_items(items, preds1)
        write_jsonl(
            tmp / "evaluation" / "discovery_primary_evaluation.jsonl", eval_rows
        )

        outcome_counts = Counter(r["outcome"] for r in eval_rows)
        pos_scores = []
        pos_labels_exact = []
        for r in eval_rows:
            if r["annotation_class"] != "readable_positive":
                continue
            if r["confidence"] is None:
                continue
            pos_scores.append(float(r["confidence"]))
            pos_labels_exact.append(1 if r["outcome"] == "exact" else 0)
        neg_scores = [
            float(r["confidence"])
            for r in eval_rows
            if r["annotation_class"] == "non_readable_negative"
            and r["confidence"] is not None
        ]
        # exact vs negative AUROC
        auroc_exact_neg = None
        if any(pos_labels_exact) and neg_scores:
            scores = []
            labels = []
            for r in eval_rows:
                if r["confidence"] is None:
                    continue
                if r["outcome"] == "exact":
                    scores.append(float(r["confidence"]))
                    labels.append(1)
                elif r["annotation_class"] == "non_readable_negative":
                    scores.append(float(r["confidence"]))
                    labels.append(0)
            auroc_exact_neg = auroc(scores, labels)
        auroc_exact_nonexact = None
        scores2, labels2 = [], []
        for r in eval_rows:
            if r["confidence"] is None:
                continue
            if r["annotation_class"] == "uncertain_excluded":
                continue
            scores2.append(float(r["confidence"]))
            labels2.append(1 if r["outcome"] == "exact" else 0)
        auroc_exact_nonexact = auroc(scores2, labels2)

        ops = build_operating_points(eval_rows)
        write_jsonl(tmp / "gate_derivation" / "operating_points.jsonl", ops)
        gate_result = derive_candidate_gate(ops)

        gate_artifact = {
            "schema_version": GATE_SCHEMA,
            "gate_status": gate_result["gate_status"],
            "threshold_selected": gate_result["threshold_selected"],
            "deployment_threshold_selected": gate_result[
                "deployment_threshold_selected"
            ],
            "selected_state": gate_result.get("selected_operating_point"),
            "unselected_state": (
                None
                if gate_result["threshold_selected"]
                else {
                    "reason": gate_result.get("reason"),
                    "nearest_diagnostic_operating_points": gate_result.get(
                        "nearest_diagnostic_operating_points"
                    ),
                }
            ),
            "candidate_confidence_cut": gate_result.get("candidate_confidence_cut"),
            "candidate_confidence_cut_exact_decimal": gate_result.get(
                "candidate_confidence_cut_exact_decimal"
            ),
            "candidate_confidence_cut_float64_hex": gate_result.get(
                "candidate_confidence_cut_float64_hex"
            ),
            "comparison_operator": ">=",
            "confidence_metric": config["preprocessing_contract"]["confidence_metric"],
            "preprocessing_contract_sha256": pp_sha,
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "model_repository_commit": repo_head,
            "discovery_annotation_freeze_listing_sha256": ann_listing,
            "canonical_manifest_sha256": sha256_file(
                canon / "discovery_primary_manifest.jsonl"
            ),
            "positive_count": 10,
            "negative_count": 27,
            "uncertain_count": 3,
            "item_level_outcome_counts": dict(outcome_counts),
            "chosen_operating_point_metrics": gate_result.get("selected_operating_point"),
            "selection_rule": gate_result.get("selection_rule")
            or config["gate_derivation"]["selection_rule"],
            "all_eligible_operating_point_ids": gate_result.get(
                "eligible_operating_point_ids", []
            ),
            "historical_threshold_reused": False,
            "discovery_reserve_opened": False,
            "holdout_opened": False,
            "target_scope": "holdout_validation_candidate_only",
            "interpretation_limits": {
                "deployment_threshold": False,
                "calibrated_probability": False,
                "permanent_accuracy_claim": False,
                "discovery_only": True,
            },
            "immutable_after_freeze": True,
            "exact_next_gate": (
                "REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE"
                if gate_result["gate_status"]
                == "DISCOVERY_SAFE_CANDIDATE_GATE_DERIVED"
                else "REBUILD-R6B_STAGE5C_DISCOVERY_PARSEQ_SIGNAL_DIAGNOSIS"
            ),
            **{
                k: gate_result[k]
                for k in (
                    "threshold_scope",
                    "threshold_source",
                    "support_count",
                    "calibrated_probability",
                    "deployment_ready",
                    "holdout_validated",
                    "mutable_after_holdout",
                    "holdout_should_not_open",
                    "discovery_reserve_should_not_open",
                    "reason",
                )
                if k in gate_result
            },
        }
        write_json(
            tmp / "gate_derivation" / "discovery_candidate_gate.json", gate_artifact
        )

        if gate_result["gate_status"] == "DISCOVERY_SAFE_CANDIDATE_GATE_DERIVED":
            final_status = "COMPLETED_DISCOVERY_PRIMARY_PARSEQ_GATE_DERIVED"
            next_gate = "REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE"
            holdout_ready = True
        else:
            final_status = "COMPLETED_DISCOVERY_PRIMARY_PARSEQ_NO_SAFE_GATE"
            next_gate = "REBUILD-R6B_STAGE5C_DISCOVERY_PARSEQ_SIGNAL_DIAGNOSIS"
            holdout_ready = False

        contract = {
            "schema_version": CONTRACT_SCHEMA,
            "project_head": head,
            "canonical_split_root": str(canon.relative_to(project_root)),
            "canonical_split_generation": config["canonical_split"]["generation"],
            "canonical_listing_sha256": listing,
            "annotation_freeze_root": str(ann_root.relative_to(project_root)),
            "annotation_freeze_listing_sha256": ann_listing,
            "input_row_contract": {
                "expected_rows": 40,
                "join_key": "split_item_id",
                "holdout_row_read": 0,
                "discovery_reserve_row_read": 0,
            },
            "preprocessing_contract": pp_contract["selected_preprocessing_contract"],
            "preprocessing_contract_sha256": pp_sha,
            "prediction_normalization_contract": pp_contract[
                "acceptance_prediction_format"
            ],
            "confidence_contract": {
                "metric": config["preprocessing_contract"]["confidence_metric"],
                "comparison_operator": ">=",
                "no_rounding_for_gate": True,
            },
            "evaluation_outcome_definitions": {
                "positive": ["exact", "wrong", "no_prediction"],
                "negative": [
                    "negative_digit_emission",
                    "negative_no_prediction",
                ],
                "uncertain": [
                    "uncertain_digit_emission",
                    "uncertain_no_prediction",
                ],
            },
            "operating_point_contract": {
                "unique_valid_emission_cuts_ascending": True,
                "sentinel_no_acceptance_excluded_from_candidate": True,
            },
            "gate_derivation_rule": config["gate_derivation"],
            "checkpoint_environment_repository_contract": {
                "checkpoint_path": str(ckpt),
                "checkpoint_sha256": config["checkpoint"]["sha256"],
                "checkpoint_bytes": int(config["checkpoint"]["byte_size"]),
                "environment": config["environment"]["name"],
                "external_repo_head": repo_head,
                "parseq_root": config["parseq_root"],
            },
            "discovery_only_scope": True,
            "holdout_reserve_exclusion_contract": {
                "holdout_opened": False,
                "discovery_reserve_opened": False,
                "holdout_crop_read_count": 0,
                "reserve_crop_read_count": 0,
            },
            "output_root": str(final_dir.relative_to(project_root)),
            "network_policy": config["network_policy"],
            "visual_budget": {"png": 0, "jpeg": 0, "mp4": 0, "contact_sheet": 0},
        }
        write_json(tmp / "discovery_parseq_contract.json", contract)

        conf_by_outcome = {
            outcome: confidence_stats(
                [r["confidence"] for r in eval_rows if r["outcome"] == outcome]
            )
            for outcome in sorted(outcome_counts)
        }
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "final_status": final_status,
            "inference_rows": 40,
            "deterministic_repeatability": det,
            "positive_outcomes": {
                "exact": outcome_counts.get("exact", 0),
                "wrong": outcome_counts.get("wrong", 0),
                "no_prediction": outcome_counts.get("no_prediction", 0),
            },
            "negative_outcomes": {
                "negative_digit_emission": outcome_counts.get(
                    "negative_digit_emission", 0
                ),
                "negative_no_prediction": outcome_counts.get(
                    "negative_no_prediction", 0
                ),
            },
            "uncertain_outcomes": {
                "uncertain_digit_emission": outcome_counts.get(
                    "uncertain_digit_emission", 0
                ),
                "uncertain_no_prediction": outcome_counts.get(
                    "uncertain_no_prediction", 0
                ),
            },
            "confidence_summaries": conf_by_outcome,
            "diagnostic_auroc": {
                "exact_vs_negative": auroc_exact_neg,
                "exact_vs_all_nonexact_excluding_uncertain": auroc_exact_nonexact,
            },
            "operating_point_count": len(ops),
            "eligible_safe_point_count": len(
                gate_result.get("eligible_operating_point_ids") or []
            ),
            "candidate_gate_result": {
                "gate_status": gate_result["gate_status"],
                "threshold_selected": gate_result["threshold_selected"],
                "candidate_confidence_cut": gate_result.get(
                    "candidate_confidence_cut"
                ),
                "comparison_operator": ">=",
                "support_count": gate_result.get("support_count"),
            },
            "warnings": [],
            "holdout_readiness": holdout_ready,
            "discovery_reserve_opened": False,
            "holdout_opened": False,
            "opencv_in_parseq_env": False,
            "opencv_required": False,
            "pil_used_for_crop_io": True,
            "next_gate": next_gate,
        }
        write_json(tmp / "discovery_parseq_summary.json", summary)

        # Media budget
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.jpg")) or list(tmp.rglob("*.mp4")):
            raise DiscoveryGateError("unexpected media in temp root")

        artifact_paths = [
            tmp / "discovery_parseq_contract.json",
            tmp / "discovery_parseq_summary.json",
            tmp / "inference" / "discovery_primary_predictions.jsonl",
            tmp / "evaluation" / "discovery_primary_evaluation.jsonl",
            tmp / "gate_derivation" / "operating_points.jsonl",
            tmp / "gate_derivation" / "discovery_candidate_gate.json",
            tmp / "runtime" / "preprocessing_contract_pre_inference.json",
            tmp / "runtime" / "discovery_inference_inputs.jsonl",
            tmp / "runtime" / "determinism_audit.json",
            tmp / "runtime" / "model_load_meta.json",
            tmp / "effective_configs" / config_path.name,
        ]
        git_after = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "artifacts": [
                {
                    "path": str(p.relative_to(tmp)),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in artifact_paths
            ],
            "input_paths": {
                "canonical_manifest": {
                    "path": str(
                        (canon / "discovery_primary_manifest.jsonl").relative_to(
                            project_root
                        )
                    ),
                    "sha256": sha256_file(canon / "discovery_primary_manifest.jsonl"),
                },
                "frozen_annotations": {
                    "path": str(
                        (
                            ann_root / "discovery_primary_annotations_frozen.csv"
                        ).relative_to(project_root)
                    ),
                    "sha256": sha256_file(
                        ann_root / "discovery_primary_annotations_frozen.csv"
                    ),
                },
            },
            "checkpoint": {
                "path": str(ckpt),
                "sha256": config["checkpoint"]["sha256"],
                "bytes": int(config["checkpoint"]["byte_size"]),
            },
            "exact_command": (
                "python scripts/run_reid_jersey_parseq_discovery_gate.py "
                f"--config {config_path} --project-root {project_root}"
            ),
            "environment_versions": {
                "python": platform.python_version(),
                "torch": __import__("torch").__version__,
                "numpy": __import__("numpy").__version__,
                "environment_name": config["environment"]["name"],
            },
            "git_head": head,
            "git_after_porcelain": git_after,
            "canonical_listing_sha_before": listing,
            "annotation_listing_sha_before": ann_listing,
            "reserve_holdout_access_audit": {
                "holdout_manifest_read": False,
                "reserve_manifest_read": False,
                "holdout_crop_read_count": 0,
                "reserve_crop_read_count": 0,
            },
            "network_audit": {
                "policy": config["network_policy"],
                "downloads": 0,
            },
            "transient_roi_files": {
                "count": 0,
                "cleanup": "n/a_in_memory_roi_only",
            },
            "atomic_finalization": True,
            "png_count": 0,
            "jpeg_count": 0,
            "mp4_count": 0,
        }
        write_json(tmp / "discovery_parseq_manifest.json", manifest)

        os.rename(tmp, final_dir)
        tmp = None
        return {
            "status": final_status,
            "output": str(final_dir),
            "next_gate": next_gate,
            "gate_status": gate_result["gate_status"],
            "candidate_cut": gate_result.get("candidate_confidence_cut"),
            "outcomes": dict(outcome_counts),
        }
    finally:
        if tmp is not None and Path(tmp).exists():
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            _PROJECT_ROOT
            / "configs/reid/jersey_parseq_discovery_gate_stage5c_rebuild_r2.yaml"
        ),
    )
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_pipeline(Path(args.config), Path(args.project_root).resolve())
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
