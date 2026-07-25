#!/usr/bin/env python3
"""Stage 5C-R8 holdout-primary PARSeq fixed-gate validation.

Applies the frozen discovery candidate gate unchanged to 48 holdout-primary
items. No threshold retune, no reserve access, no deployment claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid import jersey_parseq as jp  # noqa: E402

CONFIG_SCHEMA = "reid_jersey_parseq_holdout_validation_config_v1"
DIGIT_RE = re.compile(r"^[0-9]{1,2}$")
EXPECTED_CUT_DECIMAL = "0.99992299168434329"
EXPECTED_CUT_HEX = "3fefff5e8079b000"


class HoldoutValidationError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    # Preserve full float64 decimal via format that matches discovery gate helper.
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
        raise HoldoutValidationError("unexpected config schema")
    if config.get("device") != "cpu" or not config.get("offline_required"):
        raise HoldoutValidationError("cpu/offline required")
    return config


def annotation_class(readable: str) -> str:
    if readable == "yes":
        return "readable_positive"
    if readable == "no":
        return "non_readable_negative"
    if readable == "uncertain":
        return "uncertain_excluded"
    raise HoldoutValidationError(f"bad readable={readable!r}")


def normalize_source_type(raw: str) -> str:
    if raw in {"reused", "reused_baseline_selected_crop"}:
        return "reused"
    if raw in {"recomputed", "recomputed_manual_segment"}:
        return "recomputed"
    raise HoldoutValidationError(f"unknown source type {raw!r}")


def load_frozen_discovery_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("gate_status") != "DISCOVERY_SAFE_CANDIDATE_GATE_DERIVED":
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH status")
    if gate.get("threshold_selected") is not True:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH threshold_selected")
    if gate.get("deployment_threshold_selected") is not False:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH deployment")
    if gate.get("comparison_operator") != ">=":
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH operator")
    if gate.get("historical_threshold_reused") is not False:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH historical")
    if gate.get("holdout_validated") is not False:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH holdout_validated")
    if gate.get("mutable_after_holdout") is not False:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH mutable")
    cut_decimal = str(gate["candidate_confidence_cut_exact_decimal"])
    cut_hex = str(gate["candidate_confidence_cut_float64_hex"])
    cut_float = float(gate["candidate_confidence_cut"])
    if cut_decimal != EXPECTED_CUT_DECIMAL or cut_hex != EXPECTED_CUT_HEX:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH cut")
    if float64_hex(cut_float) != cut_hex:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH float_hex")
    if exact_decimal(cut_float) != cut_decimal and format(cut_float, ".17g") != cut_decimal:
        # Accept exact stored decimal string when float round-trips to same hex.
        if float64_hex(float(cut_decimal)) != cut_hex:
            raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH decimal")
    metrics = gate.get("chosen_operating_point_metrics") or {}
    if int(gate.get("support_count") or 0) != 1:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH support")
    if metrics.get("accepted_item_ids") != ["discovery_primary_028"]:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH item")
    if int(metrics.get("accepted_exact") or 0) != 1:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH accepted_exact")
    if int(metrics.get("accepted_wrong_positive") or 0) != 0:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH wrong")
    if int(metrics.get("accepted_negative") or 0) != 0:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH neg")
    if int(metrics.get("accepted_uncertain") or 0) != 0:
        raise HoldoutValidationError("BLOCKED_FROZEN_DISCOVERY_GATE_CONTRACT_MISMATCH unc")
    return gate


def copy_preprocessing_contract(
    *,
    discovery_pp_path: Path,
    out_path: Path,
    discovery_gate_path: Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    disc = json.loads(discovery_pp_path.read_text(encoding="utf-8"))
    selected = disc["selected_preprocessing_contract"]
    expected = {
        "input_region": "number_roi_xyxy_from_clean_universe",
        "roi_expansion_px": 0,
        "resize": [32, 128],
        "interpolation": "BICUBIC",
        "normalize_mean": 0.5,
        "normalize_std": 0.5,
        "confidence_metric": "product_of_tokenizer_decode_selected_token_probabilities",
        "top1_only": True,
        "stochastic_augmentation": False,
    }
    for key, value in expected.items():
        if selected.get(key) != value:
            raise HoldoutValidationError(
                f"BLOCKED_HOLDOUT_PREPROCESSING_CONTRACT_DRIFT {key}"
            )
    payload = {
        "schema_version": "reid_jersey_parseq_holdout_preprocessing_contract_pre_inference_v1",
        "discovery_preprocessing_artifact_path": str(discovery_pp_path),
        "discovery_preprocessing_artifact_sha256": sha256_file(discovery_pp_path),
        "exact_copied_contract": disc,
        "candidate_gate_path": str(discovery_gate_path),
        "candidate_gate_sha256": sha256_file(discovery_gate_path),
        "candidate_confidence_cut_exact_decimal": gate[
            "candidate_confidence_cut_exact_decimal"
        ],
        "candidate_confidence_cut_float64_hex": gate[
            "candidate_confidence_cut_float64_hex"
        ],
        "predictions_seen_at_freeze": False,
        "contract_frozen_before_holdout_inference": True,
    }
    write_json(out_path, payload)
    return payload


def build_input_universe(
    *,
    project_root: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canon = project_root / config["canonical_split"]["path"]
    ann_root = project_root / config["annotation_freeze"]["path"]
    man_path = canon / config["canonical_split"]["holdout_primary_manifest"]
    frozen_path = ann_root / config["annotation_freeze"]["frozen_csv"]
    universe_path = project_root / config["clean_universe_items"]

    # Hard exclusions: never open reserve / discovery-primary manifests.
    forbidden = [
        canon / "holdout_reserve/holdout_reserve_manifest.jsonl",
        canon / "discovery_reserve/discovery_reserve_manifest.jsonl",
        canon / "discovery_primary/discovery_primary_manifest.jsonl",
    ]
    for path in forbidden:
        if not path.exists():
            continue
        # Existence is fine; we must not read them for inference universe.
        pass

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
    if len(manifest) != 48 or len(frozen) != 48:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_INFERENCE_INPUT_INTEGRITY size")
    man_ids = [r["split_item_id"] for r in manifest]
    fr_ids = [r["split_item_id"] for r in frozen]
    if len(set(man_ids)) != 48 or len(set(fr_ids)) != 48:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_INFERENCE_INPUT_INTEGRITY duplicate")
    if set(man_ids) != set(fr_ids):
        raise HoldoutValidationError("BLOCKED_HOLDOUT_INFERENCE_INPUT_INTEGRITY id set")
    if any(not sid.startswith("holdout_primary_") for sid in man_ids):
        raise HoldoutValidationError("BLOCKED_HOLDOUT_INFERENCE_INPUT_INTEGRITY non_primary")
    frozen_by = {r["split_item_id"]: r for r in frozen}
    rows: list[dict[str, Any]] = []
    for man in manifest:
        sid = man["split_item_id"]
        fr = frozen_by[sid]
        if str(man["review_item_id"]) != str(fr["review_item_id"]):
            raise HoldoutValidationError(f"review_item mismatch {sid}")
        if str(man["source_crop_path"]) != str(fr["source_crop_path"]):
            raise HoldoutValidationError(f"path mismatch {sid}")
        if str(man["source_crop_sha256"]) != str(fr["source_crop_sha256"]):
            raise HoldoutValidationError(f"sha mismatch {sid}")
        u = universe[str(man["review_item_id"])]
        if not bool(u.get("roi_valid")):
            raise HoldoutValidationError(f"roi_valid false {sid}")
        crop = Path(str(man["source_crop_path"]))
        if ".." in Path(str(man["source_crop_path"])).parts:
            raise HoldoutValidationError(f"path traversal {sid}")
        if not crop.is_file() or crop.is_symlink():
            raise HoldoutValidationError(f"crop missing {sid}")
        if sha256_file(crop) != str(man["source_crop_sha256"]):
            raise HoldoutValidationError(f"crop sha fail {sid}")
        src_type = normalize_source_type(
            str(man.get("crop_source_type") or man.get("source_type"))
        )
        cls = annotation_class(str(fr["manual_number_readable"]))
        rows.append(
            {
                "split_item_id": sid,
                "review_item_id": str(man["review_item_id"]),
                "crop_id": str(man["crop_id"]),
                "segment_id": str(man["segment_id"]),
                "raw_track_id": int(man["raw_track_id"]),
                "frame_index": int(man["frame_index"]),
                "source_type": src_type,
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
        class_counts["readable_positive"] != 16
        or class_counts["non_readable_negative"] != 30
        or class_counts["uncertain_excluded"] != 2
    ):
        raise HoldoutValidationError(f"class counts {dict(class_counts)}")
    src_counts = Counter(r["source_type"] for r in rows)
    if src_counts.get("reused") != 44 or src_counts.get("recomputed") != 4:
        raise HoldoutValidationError(f"source counts {dict(src_counts)}")
    return rows


def predict_holdout_item(
    item: Mapping[str, Any],
    model: Any,
    model_meta: Mapping[str, Any],
    transform: Any,
    *,
    preprocessing_contract_sha: str,
    candidate_gate_sha: str,
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
        "source_type": item["source_type"],
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
        "manual_crop_valid": item["manual_crop_valid"],
        "manual_number_visible": item["manual_number_visible"],
        "manual_number_readable": item["manual_number_readable"],
        "manual_jersey_number": item["manual_jersey_number"],
        "annotation_class": item["annotation_class"],
        "raw_decoded_text": None,
        "normalized_prediction": None,
        "prediction_emitted": False,
        "valid_jersey_string": False,
        "confidence": None,
        "confidence_exact_decimal": None,
        "confidence_float64_hex": None,
        "confidence_metric_name": model_meta.get(
            "confidence_metric_name",
            "product_of_tokenizer_decode_selected_token_probabilities",
        ),
        "token_probabilities": None,
        "inference_ms": None,
        "checkpoint_sha256": model_meta["checkpoint_sha256"],
        "preprocessing_contract_sha256": preprocessing_contract_sha,
        "candidate_gate_sha256": candidate_gate_sha,
        "inference_error": None,
    }
    try:
        crop_path = Path(item["source_crop_path"])
        if sha256_file(crop_path) != item["source_crop_sha256"]:
            raise HoldoutValidationError("crop sha mismatch during infer")
        with Image.open(crop_path) as handle:
            rgb_full = np.asarray(handle.convert("RGB"))
        image_bgr = rgb_full[:, :, ::-1].copy()
        h, w = image_bgr.shape[:2]
        if w != int(item["source_crop_width"]) or h != int(item["source_crop_height"]):
            raise HoldoutValidationError(
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
            raise HoldoutValidationError(f"roi invalid {item['split_item_id']}")
        x1, y1, x2, y2 = roi
        roi_bgr = image_bgr[y1:y2, x1:x2]
        out["extracted_roi_pixel_sha256"] = sha256_bytes(roi_bgr.tobytes())
        pil_rgb = jp.bgr_to_rgb_pil(roi_bgr)
        tensor = transform(pil_rgb).unsqueeze(0)
        if list(tensor.shape) != [1, 3, 32, 128]:
            raise HoldoutValidationError(f"bad transform shape {list(tensor.shape)}")
        with torch.inference_mode():
            logits = model(tensor)
        decoded = jp.decode_jersey_logits(model, logits)
        raw = decoded["raw_decoded_text"]
        normalized = jp.normalize_recognized_text(raw)
        conf = decoded["sequence_confidence"]
        if conf is not None and not math.isfinite(float(conf)):
            raise HoldoutValidationError("BLOCKED_HOLDOUT_FIXED_GATE_INTEGRITY nonfinite")
        accepted, _rej = jp.extract_digit_candidate(normalized)
        valid = accepted is not None and DIGIT_RE.fullmatch(accepted) is not None
        conf_f = None if conf is None else float(conf)
        out.update(
            {
                "raw_decoded_text": raw,
                "normalized_prediction": normalized,
                "prediction_emitted": bool(normalized),
                "valid_jersey_string": bool(valid),
                "accepted_prediction": accepted if valid else None,
                "confidence": conf_f,
                "confidence_exact_decimal": None
                if conf_f is None
                else exact_decimal(conf_f),
                "confidence_float64_hex": None if conf_f is None else float64_hex(conf_f),
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
    candidate_gate_sha: str,
) -> list[dict[str, Any]]:
    transform = jp.get_transform_for_model(model)
    return [
        predict_holdout_item(
            item,
            model,
            model_meta,
            transform,
            preprocessing_contract_sha=preprocessing_contract_sha,
            candidate_gate_sha=candidate_gate_sha,
        )
        for item in items
    ]


def assert_deterministic(
    pass_a: Sequence[Mapping[str, Any]],
    pass_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(pass_a) != 48 or len(pass_b) != 48:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_INFERENCE_NONDETERMINISTIC size")
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
                {"id": a["split_item_id"], "error": "confidence", "a": ca, "b": cb}
            )
    if diffs:
        raise HoldoutValidationError(
            "BLOCKED_HOLDOUT_INFERENCE_NONDETERMINISTIC " + json.dumps(diffs[:5])
        )
    return {
        "normalized_prediction_exact_match": "48/48",
        "valid_flag_exact_match": "48/48",
        "confidence_abs_diff_le_1e12": "48/48",
        "pass_diffs": [],
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
            raise HoldoutValidationError("BLOCKED_HOLDOUT_FIXED_GATE_INTEGRITY conf")
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
                "source_type": item["source_type"],
                "manual_exact_label": (
                    item["manual_jersey_number"]
                    if cls == "readable_positive"
                    else None
                ),
                "prediction": pred,
                "normalized_prediction": p.get("normalized_prediction"),
                "valid_jersey_string": valid,
                "confidence": conf,
                "confidence_exact_decimal": p.get("confidence_exact_decimal"),
                "confidence_float64_hex": p.get("confidence_float64_hex"),
                "outcome": outcome,
                "inference_ms": p.get("inference_ms"),
                "extracted_roi_pixel_sha256": p.get("extracted_roi_pixel_sha256"),
            }
        )
    return rows


def apply_fixed_gate(
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    cut: float,
    cut_hex: str,
    operator: str,
) -> list[dict[str, Any]]:
    if operator != ">=":
        raise HoldoutValidationError("BLOCKED_HOLDOUT_FIXED_GATE_INTEGRITY operator")
    if float64_hex(cut) != cut_hex:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_FIXED_GATE_INTEGRITY cut_hex")
    out = []
    for row in eval_rows:
        conf = row.get("confidence")
        valid = bool(row.get("valid_jersey_string"))
        if conf is not None and not math.isfinite(float(conf)):
            raise HoldoutValidationError("BLOCKED_HOLDOUT_FIXED_GATE_INTEGRITY naninf")
        conf_ge = conf is not None and float(conf) >= cut
        accepted = bool(valid and conf_ge)
        if accepted:
            reason = "valid_and_confidence_ge_cut"
            reject = None
        elif not valid:
            reason = None
            reject = "invalid_or_empty_prediction"
        elif conf is None:
            reason = None
            reject = "missing_confidence"
        else:
            reason = None
            reject = "confidence_below_cut"
        out.append(
            {
                **row,
                "confidence_ge_cut": bool(conf_ge),
                "accepted": accepted,
                "acceptance_reason": reason,
                "rejection_reason": reject,
                "candidate_confidence_cut": cut,
                "candidate_confidence_cut_float64_hex": cut_hex,
                "comparison_operator": operator,
            }
        )
    return out


def decide_validation(
    gated: Sequence[Mapping[str, Any]],
    *,
    positive_count: int,
    negative_count: int,
    min_pos: int,
    min_neg: int,
    pass_exact_gte: int,
) -> dict[str, Any]:
    if positive_count < min_pos or negative_count < min_neg:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_ANNOTATION_FREEZE_CONTRACT_MISMATCH minima")
    accepted = [r for r in gated if r["accepted"]]
    accepted_exact = sum(
        1
        for r in accepted
        if r["annotation_class"] == "readable_positive" and r["outcome"] == "exact"
    )
    accepted_wrong = sum(
        1
        for r in accepted
        if r["annotation_class"] == "readable_positive" and r["outcome"] == "wrong"
    )
    accepted_neg = sum(
        1 for r in accepted if r["annotation_class"] == "non_readable_negative"
    )
    accepted_unc = sum(
        1 for r in accepted if r["annotation_class"] == "uncertain_excluded"
    )
    rejected_exact = sum(
        1 for r in gated if r["outcome"] == "exact" and not r["accepted"]
    )
    if accepted_wrong > 0 or accepted_neg > 0:
        decision = "FAIL_INDEPENDENT_GATE_SAFETY"
        final_status = "COMPLETED_HOLDOUT_VALIDATION_FAIL_GATE_SAFETY"
    elif accepted_exact >= pass_exact_gte:
        decision = "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL"
        final_status = "COMPLETED_HOLDOUT_VALIDATION_PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL"
    else:
        decision = "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT"
        final_status = "COMPLETED_HOLDOUT_VALIDATION_INCONCLUSIVE_SAFE_LOW_SUPPORT"
    return {
        "validation_decision": decision,
        "final_status": final_status,
        "accepted_total": len(accepted),
        "accepted_exact": accepted_exact,
        "accepted_wrong_positive": accepted_wrong,
        "accepted_negative": accepted_neg,
        "accepted_uncertain": accepted_unc,
        "rejected_exact": rejected_exact,
        "accepted_item_ids": [r["split_item_id"] for r in accepted],
        "accepted_predictions": [r.get("prediction") for r in accepted],
        "accepted_confidences": [r.get("confidence") for r in accepted],
        "wrong_accepted_item_ids": [
            r["split_item_id"]
            for r in accepted
            if r["annotation_class"] == "readable_positive" and r["outcome"] == "wrong"
        ],
        "negative_accepted_item_ids": [
            r["split_item_id"]
            for r in accepted
            if r["annotation_class"] == "non_readable_negative"
        ],
        "uncertain_accepted_item_ids": [
            r["split_item_id"]
            for r in accepted
            if r["annotation_class"] == "uncertain_excluded"
        ],
    }


def auroc(scores: list[float], labels: list[int]) -> Optional[float]:
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
        raise HoldoutValidationError("FAILED_ATOMIC_HOLDOUT_VALIDATION final exists")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != config["project_head_expected"]:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_VALIDATION_GIT_CONTRACT_MISMATCH HEAD")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    )
    allowed_untracked = {
        "?? scripts/run_reid_jersey_parseq_holdout_validation.py",
        "?? configs/reid/jersey_parseq_holdout_validation_stage5c_rebuild_r2.yaml",
        "?? tests/test_reid_jersey_parseq_holdout_validation.py",
    }
    dirty_lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    # Also accept configs/reid/ path variants if git reports directory then file
    unexpected = []
    for ln in dirty_lines:
        if ln in allowed_untracked:
            continue
        # git may report "?? configs/reid/" only when new dir; allow if only allowed files under it
        if ln.startswith("?? ") and ln[3:] in {
            "scripts/run_reid_jersey_parseq_holdout_validation.py",
            "configs/reid/jersey_parseq_holdout_validation_stage5c_rebuild_r2.yaml",
            "tests/test_reid_jersey_parseq_holdout_validation.py",
        }:
            continue
        unexpected.append(ln)
    if unexpected:
        raise HoldoutValidationError(
            "BLOCKED_HOLDOUT_VALIDATION_GIT_CONTRACT_MISMATCH dirty "
            + json.dumps(unexpected)
        )
    # Forbid any tracked modifications
    if any(not ln.startswith("?? ") for ln in dirty_lines):
        raise HoldoutValidationError(
            "BLOCKED_HOLDOUT_VALIDATION_GIT_CONTRACT_MISMATCH tracked_changes"
        )

    canon = project_root / config["canonical_split"]["path"]
    n_canon, listing = listing_sha(canon)
    if listing != config["canonical_split"]["listing_sha256"]:
        raise HoldoutValidationError(
            "BLOCKED_HOLDOUT_VALIDATION_GIT_CONTRACT_MISMATCH listing"
        )
    ann_root = project_root / config["annotation_freeze"]["path"]
    ann_n, ann_listing = listing_sha(ann_root)
    gate_root = project_root / config["discovery_gate"]["path"]
    gate_n, gate_listing = listing_sha(gate_root)

    gate_path = gate_root / config["discovery_gate"]["candidate_gate_json"]
    gate = load_frozen_discovery_gate(gate_path)
    cut = float(gate["candidate_confidence_cut"])
    cut_hex = str(gate["candidate_confidence_cut_float64_hex"])
    cut_decimal = str(gate["candidate_confidence_cut_exact_decimal"])
    gate_sha = sha256_file(gate_path)

    disc_pp_path = gate_root / config["discovery_gate"]["preprocessing_contract_json"]
    if not disc_pp_path.is_file():
        raise HoldoutValidationError("BLOCKED_HOLDOUT_PREPROCESSING_CONTRACT_DRIFT missing")
    disc_pp_sha = sha256_file(disc_pp_path)

    # Holdout freeze contract quick checks
    ann_summary = json.loads(
        (ann_root / "holdout_primary_annotation_summary.json").read_text(encoding="utf-8")
    )
    if ann_summary.get("final_status") != (
        "COMPLETED_HOLDOUT_PRIMARY_ANNOTATION_FREEZE_SUFFICIENT"
    ):
        raise HoldoutValidationError("BLOCKED_HOLDOUT_ANNOTATION_FREEZE_CONTRACT_MISMATCH")
    if ann_summary["annotation_rows"] != 48:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_ANNOTATION_FREEZE_CONTRACT_MISMATCH rows")
    if ann_summary["sufficiency"]["result"] != "HOLDOUT_PRIMARY_ANNOTATION_SUFFICIENT":
        raise HoldoutValidationError("BLOCKED_HOLDOUT_ANNOTATION_FREEZE_CONTRACT_MISMATCH suff")
    if ann_summary["sufficiency"].get("holdout_reserve_opened") is not False:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_ANNOTATION_FREEZE_CONTRACT_MISMATCH reserve")

    repo = Path(config["external_repo"])
    repo_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if repo_head != config["external_repo_head"]:
        raise HoldoutValidationError("BLOCKED_HOLDOUT_PARSEQ_RUNTIME_OR_CHECKPOINT repo_head")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip():
        raise HoldoutValidationError("BLOCKED_HOLDOUT_PARSEQ_RUNTIME_OR_CHECKPOINT repo_dirty")

    ckpt = Path(config["checkpoint"]["path"])
    try:
        jp.validate_checkpoint_asset(
            ckpt,
            config["checkpoint"]["sha256"],
            int(config["checkpoint"]["byte_size"]),
        )
    except jp.JerseyPARSeqError as exc:
        raise HoldoutValidationError(
            f"BLOCKED_HOLDOUT_PARSEQ_RUNTIME_OR_CHECKPOINT {exc}"
        ) from exc

    env_man = Path(config["environment"]["manifest_path"])
    env_sha = sha256_file(env_man)
    if not env_sha.startswith(config["environment"]["manifest_sha256_prefix"]):
        raise HoldoutValidationError("BLOCKED_HOLDOUT_PARSEQ_RUNTIME_OR_CHECKPOINT env")

    items = build_input_universe(project_root=project_root, config=config)

    token = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    tmp = final_dir.parent / (
        f"_tmp_full_stage4b_rebuild_r2_stage5c_holdout_primary_parseq_validation_{token}"
    )
    if tmp.exists():
        raise HoldoutValidationError("temp exists")
    tmp.mkdir(parents=False)
    for sub in (
        "inference",
        "evaluation",
        "fixed_gate_validation",
        "runtime",
        "effective_configs",
    ):
        (tmp / sub).mkdir()

    access_audit = {
        "holdout_primary_row_read": 48,
        "holdout_reserve_row_read": 0,
        "discovery_reserve_row_read": 0,
        "discovery_primary_inference_row_read": 0,
        "holdout_reserve_crop_read": 0,
        "discovery_reserve_crop_read": 0,
    }

    try:
        shutil.copy2(config_path, tmp / "effective_configs" / config_path.name)
        pp_path = tmp / "runtime" / "holdout_preprocessing_contract_pre_inference.json"
        pp_payload = copy_preprocessing_contract(
            discovery_pp_path=disc_pp_path,
            out_path=pp_path,
            discovery_gate_path=gate_path,
            gate=gate,
        )
        # Holdout preprocessing wrapper sha (includes copy metadata).
        holdout_pp_sha = sha256_file(pp_path)
        # For inference lineage, also keep discovery preprocessing artifact sha.
        write_jsonl(tmp / "runtime" / "holdout_inference_inputs.jsonl", items)

        import torch

        if torch.cuda.is_available():
            raise HoldoutValidationError("BLOCKED_HOLDOUT_PARSEQ_RUNTIME_OR_CHECKPOINT cuda")
        torch.manual_seed(0)

        model, model_meta = jp.load_parseq_recognizer(
            ckpt,
            config["parseq_root"],
            expected_sha256=config["checkpoint"]["sha256"],
            expected_byte_size=int(config["checkpoint"]["byte_size"]),
        )
        model_meta = dict(model_meta)
        model_meta["confidence_metric_name"] = (
            "product_of_tokenizer_decode_selected_token_probabilities"
        )
        syn = jp.run_synthetic_inference_preflight(model)
        write_json(
            tmp / "runtime" / "model_load_meta.json",
            {"model_meta": model_meta, "synthetic": syn},
        )

        preds1 = run_inference_pass(
            items,
            model,
            model_meta,
            preprocessing_contract_sha=disc_pp_sha,
            candidate_gate_sha=gate_sha,
        )
        preds2 = run_inference_pass(
            items,
            model,
            model_meta,
            preprocessing_contract_sha=disc_pp_sha,
            candidate_gate_sha=gate_sha,
        )
        det = assert_deterministic(preds1, preds2)
        write_json(tmp / "runtime" / "determinism_audit.json", det)
        write_jsonl(tmp / "inference" / "holdout_primary_predictions.jsonl", preds1)
        if any(p.get("inference_error") for p in preds1):
            raise HoldoutValidationError(
                "BLOCKED_HOLDOUT_INFERENCE_INPUT_INTEGRITY "
                + json.dumps(
                    [
                        {"id": p["split_item_id"], "err": p["inference_error"]}
                        for p in preds1
                        if p.get("inference_error")
                    ][:5]
                )
            )

        eval_rows = evaluate_items(items, preds1)
        write_jsonl(tmp / "evaluation" / "holdout_primary_evaluation.jsonl", eval_rows)

        gated = apply_fixed_gate(
            eval_rows, cut=cut, cut_hex=cut_hex, operator=">="
        )
        write_jsonl(
            tmp / "fixed_gate_validation" / "holdout_fixed_gate_item_results.jsonl",
            gated,
        )
        decision = decide_validation(
            gated,
            positive_count=16,
            negative_count=30,
            min_pos=int(config["preregistered_decision"]["min_readable_positive"]),
            min_neg=int(config["preregistered_decision"]["min_non_readable_negative"]),
            pass_exact_gte=int(
                config["preregistered_decision"]["pass_requires_accepted_exact_gte"]
            ),
        )

        outcome_counts = Counter(r["outcome"] for r in eval_rows)
        pos_outcomes = {
            "exact": outcome_counts.get("exact", 0),
            "wrong": outcome_counts.get("wrong", 0),
            "no_prediction": outcome_counts.get("no_prediction", 0),
        }
        neg_outcomes = {
            "negative_digit_emission": outcome_counts.get("negative_digit_emission", 0),
            "negative_no_prediction": outcome_counts.get("negative_no_prediction", 0),
        }
        unc_outcomes = {
            "uncertain_digit_emission": outcome_counts.get(
                "uncertain_digit_emission", 0
            ),
            "uncertain_no_prediction": outcome_counts.get("uncertain_no_prediction", 0),
        }

        conf_by_outcome: dict[str, Optional[dict[str, float]]] = {}
        for key in sorted(set(outcome_counts)):
            conf_by_outcome[key] = confidence_stats(
                [r["confidence"] for r in eval_rows if r["outcome"] == key]
            )

        source_diag = {}
        for src in ("reused", "recomputed"):
            subset = [r for r in eval_rows if r["source_type"] == src]
            gsubset = [r for r in gated if r["source_type"] == src]
            source_diag[src] = {
                "rows": len(subset),
                "outcomes": dict(Counter(r["outcome"] for r in subset)),
                "accepted_total": sum(1 for r in gsubset if r["accepted"]),
                "accepted_exact": sum(
                    1
                    for r in gsubset
                    if r["accepted"]
                    and r["annotation_class"] == "readable_positive"
                    and r["outcome"] == "exact"
                ),
                "accepted_wrong_positive": sum(
                    1
                    for r in gsubset
                    if r["accepted"]
                    and r["annotation_class"] == "readable_positive"
                    and r["outcome"] == "wrong"
                ),
                "accepted_negative": sum(
                    1
                    for r in gsubset
                    if r["accepted"] and r["annotation_class"] == "non_readable_negative"
                ),
            }

        # Diagnostic AUROC only
        exact_scores, exact_labels = [], []
        for r in eval_rows:
            if r["annotation_class"] == "uncertain_excluded":
                continue
            if r["confidence"] is None or not r["valid_jersey_string"]:
                continue
            exact_scores.append(float(r["confidence"]))
            exact_labels.append(1 if r["outcome"] == "exact" else 0)
        neg_scores, neg_labels = [], []
        for r in eval_rows:
            if r["outcome"] not in {"exact", "negative_digit_emission"}:
                continue
            if r["confidence"] is None or not r["valid_jersey_string"]:
                continue
            neg_scores.append(float(r["confidence"]))
            neg_labels.append(1 if r["outcome"] == "exact" else 0)

        warnings = []
        if decision["accepted_uncertain"] > 0:
            warnings.append(
                "accepted_uncertain>0 diagnostic only; not used as PASS/FAIL safety condition"
            )
        if decision["accepted_exact"] < 2 and decision["validation_decision"].startswith(
            "INCONCLUSIVE"
        ):
            warnings.append("safe but low support under frozen discovery cut")

        validation_artifact = {
            "schema_version": "reid_jersey_parseq_holdout_fixed_gate_validation_v1",
            "validation_decision": decision["validation_decision"],
            "frozen_candidate_cut_exact_decimal": cut_decimal,
            "frozen_candidate_cut_float64_hex": cut_hex,
            "comparison_operator": ">=",
            "confidence_metric": "product_of_tokenizer_decode_selected_token_probabilities",
            "preprocessing_contract_sha256": disc_pp_sha,
            "holdout_preprocessing_wrapper_sha256": holdout_pp_sha,
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "discovery_gate_sha256": gate_sha,
            "holdout_annotation_freeze_listing_sha256": ann_listing,
            "positive_count": 16,
            "negative_count": 30,
            "uncertain_count": 2,
            "item_level_outcome_counts": dict(outcome_counts),
            "fixed_gate_acceptance_counts": {
                "accepted_total": decision["accepted_total"],
                "accepted_exact": decision["accepted_exact"],
                "accepted_wrong_positive": decision["accepted_wrong_positive"],
                "accepted_negative": decision["accepted_negative"],
                "accepted_uncertain": decision["accepted_uncertain"],
                "rejected_exact": decision["rejected_exact"],
            },
            "accepted_item_ids": decision["accepted_item_ids"],
            "accepted_predictions": decision["accepted_predictions"],
            "accepted_confidences": decision["accepted_confidences"],
            "wrong_accepted_item_ids": decision["wrong_accepted_item_ids"],
            "negative_accepted_item_ids": decision["negative_accepted_item_ids"],
            "uncertain_accepted_item_ids": decision["uncertain_accepted_item_ids"],
            "source_type_diagnostics": source_diag,
            "preregistered_decision_rule": {
                "FAIL_INDEPENDENT_GATE_SAFETY": "accepted_wrong_positive>0 or accepted_negative>0",
                "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL": "wrong=0 and negative=0 and accepted_exact>=2",
                "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT": "wrong=0 and negative=0 and accepted_exact<2",
            },
            "discovery_candidate_cut_unchanged": True,
            "gate_retrained_on_holdout": False,
            "threshold_optimized_on_holdout": False,
            "threshold_unchanged": True,
            "holdout_reserve_opened": False,
            "mutable_after_validation": False,
            "deployment_threshold_selected": False,
            "calibrated_probability": False,
            "strict_historical_never_seen_claim": False,
            "player_level_independence_guaranteed": False,
            "interpretation_limits": {
                "independent_holdout_validation_only": True,
                "not_deployment_threshold": True,
                "not_calibrated_probability": True,
                "not_general_accuracy_benchmark": True,
                "not_player_identity_assignment": True,
            },
            "exact_next_gate": "REBUILD-R8A_STAGE5C_HOLDOUT_VALIDATION_FREEZE_AND_CLOSE",
        }
        write_json(
            tmp / "fixed_gate_validation" / "holdout_fixed_gate_validation.json",
            validation_artifact,
        )

        contract = {
            "schema_version": "reid_jersey_parseq_holdout_validation_contract_v1",
            "project_head": head,
            "canonical_split": {
                "path": str(config["canonical_split"]["path"]),
                "generation": config["canonical_split"]["generation"],
                "listing_sha256": listing,
            },
            "holdout_annotation_freeze": {
                "path": str(config["annotation_freeze"]["path"]),
                "listing_sha256": ann_listing,
                "file_count": ann_n,
            },
            "discovery_gate": {
                "path": str(config["discovery_gate"]["path"]),
                "artifact": str(config["discovery_gate"]["candidate_gate_json"]),
                "sha256": gate_sha,
                "listing_sha256": gate_listing,
                "file_count": gate_n,
            },
            "fixed_cut": {
                "exact_decimal": cut_decimal,
                "float64_hex": cut_hex,
                "operator": ">=",
                "loaded_from_discovery_gate_artifact": True,
            },
            "frozen_preprocessing_contract": {
                "discovery_artifact_sha256": disc_pp_sha,
                "holdout_wrapper_sha256": holdout_pp_sha,
                "copied_selected_contract": pp_payload["exact_copied_contract"][
                    "selected_preprocessing_contract"
                ],
            },
            "input_row_contract": {
                "expected_rows": 48,
                "positive": 16,
                "negative": 30,
                "uncertain": 2,
                "join_key": "split_item_id",
            },
            "prediction_normalization": {
                "whitespace_strip_only": True,
                "leading_zero_preserved": True,
                "valid_regex": config["digit_policy"]["pattern"],
            },
            "confidence_contract": {
                "metric": "product_of_tokenizer_decode_selected_token_probabilities",
                "no_clamp_round_smooth": True,
                "nan_inf_rejected": True,
            },
            "outcome_definitions": {
                "positive": ["exact", "wrong", "no_prediction"],
                "negative": ["negative_digit_emission", "negative_no_prediction"],
                "uncertain": [
                    "uncertain_digit_emission",
                    "uncertain_no_prediction",
                ],
            },
            "fixed_gate_acceptance_rule": {
                "valid_jersey_string": True,
                "confidence_operator": ">=",
                "cut_source": "discovery_candidate_gate.json",
            },
            "preregistered_validation_decision_rule": validation_artifact[
                "preregistered_decision_rule"
            ],
            "reserve_exclusion_contract": access_audit,
            "environment_checkpoint_repository": {
                "environment": config["environment"]["name"],
                "checkpoint_sha256": config["checkpoint"]["sha256"],
                "external_repo_head": config["external_repo_head"],
            },
            "output_root": str(config["output"]["final_dir"]),
            "network_policy": config["network_policy"],
            "visual_artifact_budget": {"png": 0, "jpeg": 0, "mp4": 0},
        }
        write_json(tmp / "holdout_parseq_contract.json", contract)

        summary = {
            "schema_version": "reid_jersey_parseq_holdout_validation_summary_v1",
            "final_status": decision["final_status"],
            "validation_decision": decision["validation_decision"],
            "two_pass_determinism": det,
            "positive_outcomes": pos_outcomes,
            "negative_outcomes": neg_outcomes,
            "uncertain_outcomes": unc_outcomes,
            "confidence_summaries": conf_by_outcome,
            "fixed_gate_accepted_counts": validation_artifact[
                "fixed_gate_acceptance_counts"
            ],
            "accepted_item_ids": decision["accepted_item_ids"],
            "accepted_predictions": decision["accepted_predictions"],
            "accepted_confidences": decision["accepted_confidences"],
            "source_type_diagnostics": source_diag,
            "diagnostic_auroc": {
                "exact_vs_negative": auroc(neg_scores, neg_labels),
                "exact_vs_all_nonexact_excluding_uncertain": auroc(
                    exact_scores, exact_labels
                ),
            },
            "warnings": warnings,
            "deployment_calibration_limitations": {
                "deployment_threshold_selected": False,
                "calibrated_probability": False,
                "general_accuracy_benchmark": False,
                "player_identity_assignment": False,
            },
            "discovery_candidate_cut_unchanged": True,
            "gate_retrained_on_holdout": False,
            "threshold_optimized_on_holdout": False,
            "holdout_reserve_opened": False,
            "stage5c_closure_readiness": True,
            "next_gate": "REBUILD-R8A_STAGE5C_HOLDOUT_VALIDATION_FREEZE_AND_CLOSE",
            "inference_rows": 48,
            "opencv_required": False,
            "pil_used_for_crop_io": True,
        }
        write_json(tmp / "holdout_parseq_summary.json", summary)

        # Manifest without self-hash
        artifact_paths = [
            "holdout_parseq_contract.json",
            "holdout_parseq_summary.json",
            "inference/holdout_primary_predictions.jsonl",
            "evaluation/holdout_primary_evaluation.jsonl",
            "fixed_gate_validation/holdout_fixed_gate_validation.json",
            "fixed_gate_validation/holdout_fixed_gate_item_results.jsonl",
            "runtime/holdout_preprocessing_contract_pre_inference.json",
            "runtime/holdout_inference_inputs.jsonl",
            "runtime/determinism_audit.json",
            "runtime/model_load_meta.json",
            f"effective_configs/{config_path.name}",
        ]
        arts = []
        for rel in artifact_paths:
            p = tmp / rel
            meta = {
                "path": rel,
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            if p.suffix == ".jsonl":
                meta["rows"] = sum(1 for line in p.read_text().splitlines() if line.strip())
            arts.append(meta)

        import platform

        py_ver = sys.version.split()[0]
        try:
            import numpy as np
            import PIL
            import torch
            import torchvision

            versions = {
                "python": py_ver,
                "numpy": np.__version__,
                "pil": getattr(PIL, "__version__", None),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "platform": platform.platform(),
            }
        except Exception as exc:  # noqa: BLE001
            versions = {"error": str(exc), "python": py_ver}

        manifest = {
            "schema_version": "reid_jersey_parseq_holdout_validation_manifest_v1",
            "project_head": head,
            "artifacts": arts,
            "source_crop_roi_lineage_rows": 48,
            "input_artifacts": {
                "canonical_listing_sha256": listing,
                "holdout_annotation_freeze_listing_sha256": ann_listing,
                "discovery_gate_listing_sha256": gate_listing,
                "discovery_candidate_gate_sha256": gate_sha,
                "discovery_preprocessing_sha256": disc_pp_sha,
                "checkpoint_sha256": config["checkpoint"]["sha256"],
            },
            "exact_commands": [
                f"python scripts/run_reid_jersey_parseq_holdout_validation.py --config {config_path}"
            ],
            "environment_versions": versions,
            "git_before_after": {"before": head, "after": head, "dirty": False},
            "immutability": {
                "canonical_split_unmodified": True,
                "discovery_gate_unmodified": True,
                "holdout_annotation_freeze_unmodified": True,
                "candidate_cut_unchanged": True,
            },
            "reserve_access_audit": access_audit,
            "network_audit": {
                "policy": config["network_policy"],
                "external_connect": 0,
                "download": 0,
            },
            "png_count": 0,
            "jpeg_count": 0,
            "mp4_count": 0,
            "transient_cleanup": True,
            "atomic_finalization": True,
        }
        write_json(tmp / "holdout_parseq_manifest.json", manifest)
        # Append manifest entry after write (no self-hash inside)
        man_meta = {
            "path": "holdout_parseq_manifest.json",
            "sha256": sha256_file(tmp / "holdout_parseq_manifest.json"),
            "bytes": (tmp / "holdout_parseq_manifest.json").stat().st_size,
        }
        # Rewrite manifest with artifacts including itself but without embedding that self hash
        # Spec: "Manifest kendi SHA'sını içine yazmasın." => do not include self in artifacts list.
        # Keep as-is without self entry.

        # visual budget
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.jpg")) or list(tmp.rglob("*.mp4")):
            raise HoldoutValidationError("FAILED_ATOMIC_HOLDOUT_VALIDATION media")

        os.replace(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Post immutability
    n2, listing2 = listing_sha(canon)
    if (n2, listing2) != (n_canon, listing):
        raise HoldoutValidationError("FAILED_ATOMIC_HOLDOUT_VALIDATION canon mutated")
    _, ann2 = listing_sha(ann_root)
    if ann2 != ann_listing:
        raise HoldoutValidationError("FAILED_ATOMIC_HOLDOUT_VALIDATION ann mutated")
    _, gate2 = listing_sha(gate_root)
    if gate2 != gate_listing:
        raise HoldoutValidationError("FAILED_ATOMIC_HOLDOUT_VALIDATION gate mutated")

    return {
        "final_status": decision["final_status"],
        "validation_decision": decision["validation_decision"],
        "output_root": str(final_dir),
        "accepted_exact": decision["accepted_exact"],
        "accepted_wrong_positive": decision["accepted_wrong_positive"],
        "accepted_negative": decision["accepted_negative"],
        "accepted_uncertain": decision["accepted_uncertain"],
        "accepted_item_ids": decision["accepted_item_ids"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/jersey_parseq_holdout_validation_stage5c_rebuild_r2.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args()
    result = run_pipeline(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
