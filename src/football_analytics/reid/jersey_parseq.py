"""Offline CPU SoccerNet-finetuned PARSeq recognizer-only smoke adapter.

Stage 5C-C3D:
- bundled ``str/parseq`` via PYTHONPATH (no root ``str.py`` import);
- local SoccerNet-finetuned ``.ckpt`` only (no generic ``parseq-bb5792a6.pt``);
- Stage 5A number-search ROI reuse with BGR→RGB;
- jersey decode slice ``logits[:, :3, :11]`` (static ``str.py`` reference);
- digit-only ``^[0-9]{1,2}$``; no threshold; manual-label blind inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
import re
import statistics
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from football_analytics.reid import jersey_mmocr as jm

ADAPTER_VERSION = "reid_jersey_parseq_smoke_adapter_v1"
PREDICTION_SCHEMA_VERSION = "reid_jersey_parseq_prediction_v1"
ITEM_EVALUATION_SCHEMA_VERSION = "reid_jersey_parseq_item_evaluation_v1"
RESULTS_SUMMARY_SCHEMA_VERSION = "reid_jersey_parseq_smoke_results_summary_v1"
RUNTIME_SUMMARY_SCHEMA_VERSION = "reid_jersey_parseq_smoke_runtime_summary_v1"
RUN_MANIFEST_SCHEMA_VERSION = "reid_jersey_parseq_smoke_run_manifest_v1"
CHECKPOINT_CONTRACT_SCHEMA = "reid_jersey_parseq_checkpoint_contract_v1"
STATIC_AUDIT_SCHEMA = "reid_jersey_parseq_static_checkpoint_audit_v1"

POSITIVE_CLASS = jm.POSITIVE_CLASS
ALL_CLASSES = jm.ALL_CLASSES
DIGIT_PATTERN = jm.DIGIT_PATTERN
BLIND_FORBIDDEN_FIELDS = jm.BLIND_FORBIDDEN_FIELDS | frozenset(
    {
        "exact_match",
        "exact match",
        "identity",
        "global_id",
        "global-ID",
        "team",
        "expected_label",
        "manual_visible",
        "manual_readable",
    }
)

# Jersey decode overlay from jersey-number-pipeline ``str.py`` (static reference).
JERSEY_DECODE_SEQ_LEN = 3
JERSEY_DECODE_CLASS_COUNT = 11
JERSEY_DIGIT_PREFIX = "0123456789"

GENERIC_PRETRAINED_NAME = "parseq-bb5792a6.pt"

FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "manual_jersey_number",
        "manual_visible",
        "manual_readable",
        "manual_number_visible",
        "manual_number_readable",
        "expected_label",
        "exact_match",
        "identity",
        "global_id",
        "team_id",
        "gallery_id",
    }
)


class JerseyPARSeqError(RuntimeError):
    """Contract failure for the PARSeq smoke adapter."""


def sha256_file(path: str | Path) -> str:
    return jm.sha256_file(path)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return jm.load_jsonl(path)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    jm.write_jsonl(path, rows)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    jm.write_json(path, payload)


def normalize_recognized_text(raw_text: Optional[str]) -> str:
    return jm.normalize_recognized_text(raw_text)


def extract_digit_candidate(normalized_text: str) -> tuple[Optional[str], Optional[str]]:
    return jm.extract_digit_candidate(normalized_text)


def clamp_roi(
    x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int
) -> Optional[tuple[int, int, int, int]]:
    return jm.clamp_roi(x_min, y_min, x_max, y_max, width, height)


def assert_blind_records_safe(records: Sequence[Mapping[str, Any]]) -> None:
    jm.assert_blind_records_safe(records)
    for row in records:
        leaked = sorted(FORBIDDEN_PREDICTION_KEYS.intersection(row.keys()))
        if leaked:
            raise JerseyPARSeqError(f"blind record contains forbidden fields: {leaked}")


def assert_prediction_blind(prediction: Mapping[str, Any]) -> None:
    leaked = sorted(FORBIDDEN_PREDICTION_KEYS.intersection(prediction.keys()))
    if leaked:
        raise JerseyPARSeqError(f"prediction contains forbidden fields: {leaked}")


def validate_bundled_parseq_root(parseq_root: str | Path) -> Path:
    root = Path(parseq_root).resolve()
    if not root.is_dir():
        raise JerseyPARSeqError(f"PARSEQ_ROOT missing: {root}")
    for rel in (
        "strhub/__init__.py",
        "strhub/models/utils.py",
        "strhub/models/parseq/system.py",
        "strhub/data/module.py",
        "strhub/data/utils.py",
    ):
        if not (root / rel).is_file():
            raise JerseyPARSeqError(f"bundled PARSeq missing {rel} under {root}")
    # Root str.py must exist as sibling of str/ but must not be imported.
    repo_root = root.parent.parent
    str_py = repo_root / "str.py"
    if not str_py.is_file():
        raise JerseyPARSeqError(f"expected repo str.py at {str_py} (must remain unimported)")
    return root


def ensure_module_from_bundled(module: Any, parseq_root: Path, label: str) -> Path:
    path = Path(getattr(module, "__file__", "")).resolve()
    root = parseq_root.resolve()
    if not str(path).startswith(str(root)):
        raise JerseyPARSeqError(f"{label} not loaded from bundled PARSeq: {path}")
    if "site-packages" in str(path):
        raise JerseyPARSeqError(f"{label} resolved to site-packages: {path}")
    return path


def validate_checkpoint_asset(
    checkpoint_path: str | Path,
    expected_sha256: str,
    expected_byte_size: int,
) -> Path:
    path = Path(checkpoint_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise JerseyPARSeqError(f"checkpoint must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size != int(expected_byte_size):
        raise JerseyPARSeqError(f"checkpoint size mismatch: {size} != {expected_byte_size}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise JerseyPARSeqError(f"checkpoint sha256 mismatch: {digest}")
    if path.name == GENERIC_PRETRAINED_NAME:
        raise JerseyPARSeqError("generic pretrained weight is forbidden for C3D")
    return path


def static_checkpoint_audit(checkpoint_path: str | Path) -> dict[str, Any]:
    """ZIP + pickletools GLOBAL inventory without unpickling objects."""
    path = Path(checkpoint_path)
    if not zipfile.is_zipfile(path):
        raise JerseyPARSeqError("BLOCKED_CHECKPOINT_STATIC_AUDIT: not a zipfile")
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        abs_paths = [n for n in names if n.startswith("/") or re.match(r"^[A-Za-z]:", n)]
        trav = [n for n in names if ".." in Path(n).parts]
        dups = [n for n, count in Counter(names).items() if count > 1]
        enc = [info.filename for info in infos if info.flag_bits & 0x1]
        exec_ext = {
            n
            for n in names
            if Path(n).suffix.lower() in {".py", ".sh", ".exe", ".so", ".dll", ".bat", ".ps1"}
        }
        ratios = []
        for info in infos:
            if info.compress_size > 0 and info.file_size / info.compress_size > 100:
                ratios.append(
                    {
                        "name": info.filename,
                        "ratio": info.file_size / info.compress_size,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                    }
                )
        if "archive/data.pkl" not in names:
            raise JerseyPARSeqError("BLOCKED_CHECKPOINT_STATIC_AUDIT: missing archive/data.pkl")
        data = zf.read("archive/data.pkl")
    buf = io.StringIO()
    pickletools.dis(data, out=buf)
    dis_text = buf.getvalue()
    raw_globals = re.findall(r"GLOBAL\s+'([^']+)'", dis_text)
    refs: list[tuple[str, str]] = []
    for item in raw_globals:
        parts = item.split(" ", 1)
        refs.append((parts[0], parts[1] if len(parts) == 2 else ""))
    unique = sorted(set(refs))
    allowed = (
        "builtins",
        "__builtin__",
        "__builtins__",
        "collections",
        "torch",
        "pytorch_lightning",
        "argparse",
        "numpy",
        "typing",
        "copyreg",
        "_codecs",
        "functools",
    )
    suspicious = []
    for module_name, attr in unique:
        ok = any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in allowed)
        if not ok:
            suspicious.append({"module": module_name, "name": attr})
    if abs_paths or trav or dups or enc or exec_ext or ratios or suspicious:
        raise JerseyPARSeqError(
            "BLOCKED_CHECKPOINT_STATIC_AUDIT: "
            + json.dumps(
                {
                    "absolute_paths": abs_paths,
                    "path_traversal": trav,
                    "duplicates": dups,
                    "encrypted": enc,
                    "exec_ext": sorted(exec_ext),
                    "ratios": ratios,
                    "suspicious": suspicious,
                }
            )
        )
    return {
        "schema": STATIC_AUDIT_SCHEMA,
        "is_zipfile": True,
        "member_count": len(names),
        "compressed_total": sum(info.compress_size for info in infos),
        "uncompressed_total": sum(info.file_size for info in infos),
        "absolute_paths": [],
        "path_traversal": [],
        "duplicate_members": [],
        "encrypted_members": [],
        "executable_extensions": [],
        "extreme_compression_ratios": [],
        "has_data_pkl": True,
        "data_pkl_bytes": len(data),
        "pickle_protocol": 2,
        "global_refs": [{"module": m, "name": n} for m, n in unique],
        "modules": sorted({m for m, _ in unique}),
        "suspicious_refs": [],
        "status": "pass",
        "residual_risk": (
            "trusted_source_checkpoint_pickle_deserialization_risk_accepted_for_local_offline_research_smoke"
        ),
        "official_checksum_available": False,
        "official_checksum": None,
        "source": "official_repo_linked_google_drive_asset",
        "first_40_members": names[:40],
    }


def bgr_to_rgb_pil(roi_bgr: Any) -> Any:
    """Convert OpenCV BGR ndarray to PIL RGB (no other preprocessing)."""
    from PIL import Image

    if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
        raise JerseyPARSeqError("empty ROI")
    if len(roi_bgr.shape) != 3 or roi_bgr.shape[2] != 3:
        raise JerseyPARSeqError(f"ROI must be HxWx3 BGR, got {getattr(roi_bgr, 'shape', None)}")
    rgb = roi_bgr[:, :, ::-1].copy()
    return Image.fromarray(rgb, mode="RGB")


def product_confidence(token_probabilities: Sequence[float]) -> Optional[float]:
    if not token_probabilities:
        return None
    value = 1.0
    for prob in token_probabilities:
        value *= float(prob)
    if not math.isfinite(value):
        return None
    return float(value)


def decode_jersey_logits(model: Any, logits: Any) -> dict[str, Any]:
    """Apply static str.py jersey slice then bundled tokenizer.decode."""
    import torch

    if logits.ndim != 3:
        raise JerseyPARSeqError(f"logits rank must be 3, got {tuple(logits.shape)}")
    _batch, seq_len, class_count = logits.shape
    if seq_len < JERSEY_DECODE_SEQ_LEN or class_count < JERSEY_DECODE_CLASS_COUNT:
        raise JerseyPARSeqError(
            "BLOCKED_CHECKPOINT_MODEL_CONTRACT: logits too small for jersey slice "
            f"{tuple(logits.shape)}"
        )
    sliced = logits[:, :JERSEY_DECODE_SEQ_LEN, :JERSEY_DECODE_CLASS_COUNT]
    if not bool(torch.isfinite(sliced).all()):
        raise JerseyPARSeqError("non-finite logits")
    probs = sliced.softmax(-1)
    preds, token_prob_tensors = model.tokenizer.decode(probs)
    raw_text = preds[0] if preds else ""
    token_probs = [float(x) for x in token_prob_tensors[0].detach().cpu().tolist()]
    token_ids = probs[0].argmax(-1).detach().cpu().tolist()
    eos_id = int(model.tokenizer.eos_id)
    try:
        eos_position = token_ids.index(eos_id)
    except ValueError:
        eos_position = None
    return {
        "raw_output_shape": list(logits.shape),
        "sliced_shape": list(sliced.shape),
        "raw_decoded_text": raw_text,
        "token_ids": [int(x) for x in token_ids],
        "token_probabilities": token_probs,
        "eos_position": eos_position,
        "sequence_confidence": product_confidence(token_probs),
        "confidence_method": "product_of_tokenizer_decode_selected_token_probabilities",
        "decode_method": "str_py_logits_slice_3x11_softmax_tokenizer_decode",
    }


def validate_runtime_jersey_contract(model: Any) -> dict[str, Any]:
    hp = model.hparams
    charset_train = str(getattr(hp, "charset_train", ""))
    charset_test = str(getattr(hp, "charset_test", ""))
    max_label_length = int(getattr(hp, "max_label_length"))
    img_size = [int(x) for x in list(getattr(hp, "img_size"))]
    if type(model).__name__ != "PARSeq":
        raise JerseyPARSeqError("BLOCKED_CHECKPOINT_MODEL_CONTRACT: model class is not PARSeq")
    if img_size != [32, 128]:
        raise JerseyPARSeqError(f"BLOCKED_CHECKPOINT_MODEL_CONTRACT: img_size={img_size}")
    if not charset_train.startswith(JERSEY_DIGIT_PREFIX):
        raise JerseyPARSeqError(
            "BLOCKED_CHECKPOINT_MODEL_CONTRACT: charset_train does not start with digits"
        )
    head_classes = int(model.head.out_features) if hasattr(model, "head") else None
    if head_classes is None or head_classes < JERSEY_DECODE_CLASS_COUNT:
        raise JerseyPARSeqError(
            f"BLOCKED_CHECKPOINT_MODEL_CONTRACT: head classes={head_classes}"
        )
    # Disclose pickled hparams vs jersey decode overlay (do not rewrite hparams).
    return {
        "architecture": "PARSeq",
        "pickled_charset_train": charset_train,
        "pickled_charset_test": charset_test,
        "pickled_max_label_length": max_label_length,
        "img_size": img_size,
        "patch_size": [int(x) for x in list(getattr(hp, "patch_size", []))],
        "embed_dim": int(getattr(hp, "embed_dim")),
        "tokenizer_len": len(model.tokenizer),
        "eos_id": int(model.tokenizer.eos_id),
        "bos_id": int(model.tokenizer.bos_id),
        "pad_id": int(model.tokenizer.pad_id),
        "head_out_features": head_classes,
        "jersey_decode_seq_len": JERSEY_DECODE_SEQ_LEN,
        "jersey_decode_class_count": JERSEY_DECODE_CLASS_COUNT,
        "jersey_decode_classes_are_eos_plus_digits": charset_train.startswith(JERSEY_DIGIT_PREFIX),
        "naive_hparams_match_digits_and_max_len_2": (
            charset_test == JERSEY_DIGIT_PREFIX and max_label_length == 2
        ),
        "runtime_decode_contract": "str_py_logits_slice_3x11",
        "contract_status": "pass_jersey_decode_overlay_disclosed_pickled_hparams",
    }


def load_parseq_recognizer(
    checkpoint_path: str | Path,
    parseq_root: str | Path,
    *,
    expected_sha256: str,
    expected_byte_size: int,
) -> tuple[Any, dict[str, Any]]:
    """Load SoccerNet-finetuned PARSeq on CPU from bundled source only."""
    import sys

    import torch

    root = validate_bundled_parseq_root(parseq_root)
    ckpt = validate_checkpoint_asset(checkpoint_path, expected_sha256, expected_byte_size)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if torch.cuda.is_available():
        raise JerseyPARSeqError("CUDA must be unavailable")

    import strhub
    import strhub.data.module as data_module
    import strhub.data.utils as data_utils
    import strhub.models.parseq.system as parseq_system
    import strhub.models.utils as model_utils

    origins = {
        "strhub": str(ensure_module_from_bundled(strhub, root, "strhub")),
        "strhub.models.utils": str(ensure_module_from_bundled(model_utils, root, "utils")),
        "strhub.models.parseq.system": str(
            ensure_module_from_bundled(parseq_system, root, "PARSeq system")
        ),
        "strhub.data.module": str(ensure_module_from_bundled(data_module, root, "data.module")),
        "strhub.data.utils": str(ensure_module_from_bundled(data_utils, root, "data.utils")),
    }
    if "str" in sys.modules and getattr(sys.modules["str"], "__file__", None):
        # stdlib str has no __file__; a loaded repo str.py would be catastrophic.
        loaded = sys.modules["str"]
        if getattr(loaded, "__file__", None):
            raise JerseyPARSeqError("root str.py must not be imported")

    t0 = time.perf_counter()
    # Official pipeline may pass charset_test=digits; we do not override pickled
    # max_label_length. Decode uses the static str.py slice instead.
    model = model_utils.load_from_checkpoint(str(ckpt), map_location="cpu")
    model = model.eval().to("cpu")
    load_s = time.perf_counter() - t0
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise JerseyPARSeqError("model parameters not all on CPU")
    contract = validate_runtime_jersey_contract(model)
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    meta = {
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": expected_sha256,
        "checkpoint_byte_size": int(expected_byte_size),
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "model_device": "cpu",
        "model_dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
        "parameter_count": int(param_count),
        "trainable_parameter_count": int(trainable),
        "load_seconds": load_s,
        "bundled_module_origins": origins,
        "generic_pretrained_fallback_used": False,
        "load_from_checkpoint": "strhub.models.utils.load_from_checkpoint",
        "map_location": "cpu",
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_load": True,
        "runtime_contract": contract,
        "adapter_version": ADAPTER_VERSION,
    }
    return model, meta


def get_transform_for_model(model: Any) -> Any:
    from strhub.data.module import SceneTextDataModule

    img_size = tuple(int(x) for x in model.hparams.img_size)
    return SceneTextDataModule.get_transform(img_size, augment=False, rotation=0)


def run_synthetic_inference_preflight(model: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    transform = get_transform_for_model(model)
    arr = np.zeros((48, 96, 3), dtype=np.uint8)
    arr[8:40, 16:80] = [30, 90, 180]
    image = Image.fromarray(arr, mode="RGB")
    tensor = transform(image).unsqueeze(0)
    expected = [1, 3, int(model.hparams.img_size[0]), int(model.hparams.img_size[1])]
    if list(tensor.shape) != expected:
        raise JerseyPARSeqError(f"BLOCKED_SYNTHETIC_INFERENCE shape {list(tensor.shape)}")
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise JerseyPARSeqError("BLOCKED_SYNTHETIC_INFERENCE dtype/device")
    with torch.inference_mode():
        logits = model(tensor)
    decoded = decode_jersey_logits(model, logits)
    return {
        "status": "pass",
        "input_shape": list(tensor.shape),
        "logits_shape": decoded["raw_output_shape"],
        "raw_decoded_text": decoded["raw_decoded_text"],
        "finite": True,
        "included_in_metrics": False,
    }


def predict_single_item(
    item: Mapping[str, Any],
    model: Any,
    model_meta: Mapping[str, Any],
    transform: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    prediction: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "review_item_id": item["review_item_id"],
        "pilot_index": int(item["pilot_index"]),
        "selection_class": item["selection_class"],
        "crop_id": item["crop_id"],
        "segment_id": item["segment_id"],
        "raw_track_id": int(item["raw_track_id"]),
        "frame_index": int(item["frame_index"]),
        "source_crop_path": item["source_crop_path"],
        "source_crop_sha256": item["source_crop_sha256"],
        "original_roi_xyxy": None,
        "original_roi_width": None,
        "original_roi_height": None,
        "transformed_shape": None,
        "color_conversion": "bgr_to_rgb",
        "interpolation": "BICUBIC",
        "normalization_mean": 0.5,
        "normalization_std": 0.5,
        "checkpoint_path": model_meta["checkpoint_path"],
        "checkpoint_sha256": model_meta["checkpoint_sha256"],
        "checkpoint_byte_size": model_meta["checkpoint_byte_size"],
        "model_class": model_meta["model_class"],
        "model_device": model_meta["model_device"],
        "model_dtype": model_meta["model_dtype"],
        "raw_output_shape": None,
        "token_ids": None,
        "token_probabilities": None,
        "eos_position": None,
        "raw_decoded_text": None,
        "normalized_text": None,
        "accepted_prediction": None,
        "rejection_reason": None,
        "sequence_confidence": None,
        "confidence_method": None,
        "inference_ms": None,
        "inference_error": None,
        "warning_codes": [],
        "adapter_version": ADAPTER_VERSION,
        "decode_method": None,
    }
    t0 = time.perf_counter()
    try:
        crop_path = Path(item["source_crop_path"])
        if not crop_path.is_file() or crop_path.is_symlink():
            raise JerseyPARSeqError(f"source crop missing/symlink: {crop_path}")
        actual_sha = sha256_file(crop_path)
        if actual_sha != item["source_crop_sha256"]:
            raise JerseyPARSeqError(
                f"crop SHA mismatch for {item['review_item_id']}: {actual_sha}"
            )
        # Pillow loads RGB; convert to BGR ndarray to preserve the BGR→RGB adapter
        # contract without requiring OpenCV in sn-jersey-parseq-cpu.
        with Image.open(crop_path) as handle:
            rgb_full = np.asarray(handle.convert("RGB"))
        image_bgr = rgb_full[:, :, ::-1].copy()
        height, width = image_bgr.shape[:2]
        roi = clamp_roi(
            int(item["roi_x_min"]),
            int(item["roi_y_min"]),
            int(item["roi_x_max"]),
            int(item["roi_y_max"]),
            width,
            height,
        )
        if roi is None:
            prediction["rejection_reason"] = "roi_invalid"
            prediction["inference_ms"] = (time.perf_counter() - t0) * 1000.0
            assert_prediction_blind(prediction)
            return prediction
        x1, y1, x2, y2 = roi
        prediction["original_roi_xyxy"] = [x1, y1, x2, y2]
        prediction["original_roi_width"] = x2 - x1
        prediction["original_roi_height"] = y2 - y1
        roi_bgr = image_bgr[y1:y2, x1:x2]
        pil_rgb = bgr_to_rgb_pil(roi_bgr)
        tensor = transform(pil_rgb).unsqueeze(0)
        prediction["transformed_shape"] = list(tensor.shape)
        if list(tensor.shape) != [1, 3, 32, 128]:
            raise JerseyPARSeqError(f"unexpected transformed shape {list(tensor.shape)}")
        with torch.inference_mode():
            logits = model(tensor)
        decoded = decode_jersey_logits(model, logits)
        prediction.update(
            {
                "raw_output_shape": decoded["raw_output_shape"],
                "token_ids": decoded["token_ids"],
                "token_probabilities": decoded["token_probabilities"],
                "eos_position": decoded["eos_position"],
                "raw_decoded_text": decoded["raw_decoded_text"],
                "sequence_confidence": decoded["sequence_confidence"],
                "confidence_method": decoded["confidence_method"],
                "decode_method": decoded["decode_method"],
            }
        )
        normalized = normalize_recognized_text(decoded["raw_decoded_text"])
        prediction["normalized_text"] = normalized
        accepted, rejection = extract_digit_candidate(normalized)
        prediction["accepted_prediction"] = accepted
        prediction["rejection_reason"] = rejection
    except Exception as exc:  # noqa: BLE001 - per-item isolation
        prediction["inference_error"] = f"{type(exc).__name__}: {exc}"
        prediction["warning_codes"] = ["inference_exception"]
    prediction["inference_ms"] = (time.perf_counter() - t0) * 1000.0
    assert_prediction_blind(prediction)
    return prediction


def run_blind_inference(
    items: Sequence[Mapping[str, Any]],
    model: Any,
    model_meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    assert_blind_records_safe(items)
    transform = get_transform_for_model(model)
    predictions = [predict_single_item(item, model, model_meta, transform) for item in items]
    if [p["review_item_id"] for p in predictions] != [i["review_item_id"] for i in items]:
        raise JerseyPARSeqError("prediction order diverged from blind manifest")
    return predictions


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    expected_class_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(predictions) != len(reference_rows):
        raise JerseyPARSeqError(
            f"prediction/reference count mismatch: {len(predictions)} vs {len(reference_rows)}"
        )
    reference_by_id = {row["review_item_id"]: row for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise JerseyPARSeqError("duplicate review_item_id in evaluation reference")

    pred_ids = [p["review_item_id"] for p in predictions]
    ref_ids = [r["review_item_id"] for r in reference_rows]
    if sorted(pred_ids) != sorted(ref_ids):
        missing = sorted(set(ref_ids) - set(pred_ids))
        extra = sorted(set(pred_ids) - set(ref_ids))
        raise JerseyPARSeqError(f"join mismatch missing={missing} extra={extra}")
    if len(pred_ids) != len(set(pred_ids)):
        raise JerseyPARSeqError("duplicate prediction review_item_id")

    item_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        # Join uses ids; order follows predictions (blind manifest order).
        reference = reference_by_id[prediction["review_item_id"]]
        if int(reference["pilot_index"]) != int(prediction["pilot_index"]):
            raise JerseyPARSeqError(f"pilot_index mismatch for {prediction['review_item_id']}")
        if reference["selection_class"] != prediction["selection_class"]:
            raise JerseyPARSeqError(f"selection_class mismatch for {prediction['review_item_id']}")
        ref_sha = reference.get("source_crop_sha256")
        if ref_sha is not None and ref_sha != prediction["source_crop_sha256"]:
            raise JerseyPARSeqError(f"crop SHA mismatch on join for {prediction['review_item_id']}")

        predicted = prediction.get("accepted_prediction")
        error = prediction.get("inference_error")
        selection_class = prediction["selection_class"]
        if selection_class == POSITIVE_CLASS:
            expected = (reference.get("manual_jersey_number") or "").strip()
            if error is not None:
                outcome = "inference_error"
            elif predicted is None:
                outcome = "no_prediction"
            elif predicted == expected:
                outcome = "exact_match"
            else:
                outcome = "wrong_number"
        else:
            if error is not None:
                outcome = "inference_error"
            elif predicted is not None:
                outcome = "number_emitted"
            else:
                outcome = "rejected"
        item_rows.append(
            {
                "schema_version": ITEM_EVALUATION_SCHEMA_VERSION,
                "review_item_id": prediction["review_item_id"],
                "pilot_index": prediction["pilot_index"],
                "selection_class": selection_class,
                "source_crop_sha256": prediction["source_crop_sha256"],
                "outcome": outcome,
                "accepted_prediction": predicted,
                "raw_decoded_text": prediction.get("raw_decoded_text"),
                "normalized_text": prediction.get("normalized_text"),
                "manual_jersey_number": reference.get("manual_jersey_number"),
                "manual_digit_count": reference.get("manual_digit_count"),
                "sequence_confidence": prediction.get("sequence_confidence"),
                "rejection_reason": prediction.get("rejection_reason"),
                "inference_error": error,
                "inference_ms": prediction.get("inference_ms"),
            }
        )

    summary = build_results_summary(item_rows, predictions, expected_class_counts)
    return item_rows, summary


def _confidence_stats(values: Sequence[Optional[float]]) -> Optional[dict[str, float]]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
    }


def _distribution(values: Sequence[Optional[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value) if value is not None else "null"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_evidence_and_decision(
    positive_exact: int,
    positive_accepted: int,
    negative_emission: int,
    runtime_ok: bool,
) -> tuple[list[str], str]:
    labels: list[str] = []
    if positive_exact > 0:
        labels.append("PARSEQ_EXACT_SIGNAL_PRESENT")
        labels.append("CURRENT_STAGE5A_ROI_HAS_SOME_PARSEQ_COMPATIBILITY_SIGNAL")
    else:
        labels.append("NO_PARSEQ_EXACT_SIGNAL_IN_FROZEN_ROI_SMOKE")
        labels.append("CURRENT_STAGE5A_ROI_COMPATIBILITY_NOT_DEMONSTRATED")
    if negative_emission > 0:
        labels.append("PARSEQ_NEGATIVE_EMISSION_RISK_PRESENT")
    else:
        labels.append("NO_NEGATIVE_DIGIT_EMISSION_OBSERVED")
    if runtime_ok:
        labels.append("PARSEQ_CHECKPOINT_RUNTIME_CONTRACT_VALIDATED")

    if positive_exact > 0 and negative_emission == 0:
        decision = "GO_STAGE5C_C3E_PARSEQ_SIGNAL_FEASIBILITY"
    elif positive_exact > 0 and negative_emission > 0:
        decision = "GO_STAGE5C_C3E_PARSEQ_FALSE_POSITIVE_AUDIT"
    elif positive_exact == 0 and positive_accepted > 0:
        decision = "GO_STAGE5C_C3E_PARSEQ_INPUT_COMPATIBILITY_AUDIT"
    else:
        decision = "CLOSE_CURRENT_PARSEQ_ROI_SMOKE_PATH"
    return labels, decision


def build_results_summary(
    item_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    expected_class_counts: Mapping[str, int],
    *,
    checkpoint_meta: Optional[Mapping[str, Any]] = None,
    c1_metrics: Optional[Mapping[str, Any]] = None,
    c2_note: Optional[str] = None,
) -> dict[str, Any]:
    positive_rows = [row for row in item_rows if row["selection_class"] == POSITIVE_CLASS]
    negative_rows = [row for row in item_rows if row["selection_class"] != POSITIVE_CLASS]
    if len(positive_rows) != int(expected_class_counts[POSITIVE_CLASS]):
        raise JerseyPARSeqError("positive count mismatch")
    if len(item_rows) != sum(int(v) for v in expected_class_counts.values()):
        raise JerseyPARSeqError("total class count mismatch")

    exact = sum(1 for row in positive_rows if row["outcome"] == "exact_match")
    wrong = sum(1 for row in positive_rows if row["outcome"] == "wrong_number")
    none = sum(1 for row in positive_rows if row["outcome"] == "no_prediction")
    pos_err = sum(1 for row in positive_rows if row["outcome"] == "inference_error")
    if exact + wrong + none + pos_err != len(positive_rows):
        raise JerseyPARSeqError("positive outcome partition failed")
    if pos_err == 0 and exact + wrong + none != 20:
        raise JerseyPARSeqError("exact+wrong+no_prediction must equal 20")

    positive_accepted = sum(1 for row in positive_rows if row["accepted_prediction"] is not None)
    negative_emission = sum(1 for row in negative_rows if row["outcome"] == "number_emitted")
    negative_rejected = sum(1 for row in negative_rows if row["outcome"] == "rejected")

    per_jersey: dict[str, dict[str, Any]] = {}
    for row in positive_rows:
        jersey = str(row.get("manual_jersey_number") or "")
        entry = per_jersey.setdefault(
            jersey,
            {
                "item_count": 0,
                "exact_match_count": 0,
                "wrong_number_count": 0,
                "no_prediction_count": 0,
                "predictions": [],
            },
        )
        entry["item_count"] += 1
        entry["predictions"].append(row.get("accepted_prediction"))
        if row["outcome"] == "exact_match":
            entry["exact_match_count"] += 1
        elif row["outcome"] == "wrong_number":
            entry["wrong_number_count"] += 1
        elif row["outcome"] == "no_prediction":
            entry["no_prediction_count"] += 1

    one_digit_ref = [
        row for row in positive_rows if str(row.get("manual_digit_count") or "") == "1"
    ]
    two_digit_ref = [
        row for row in positive_rows if str(row.get("manual_digit_count") or "") == "2"
    ]

    def _pos_subset_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "item_count": len(rows),
            "exact_match_count": sum(1 for r in rows if r["outcome"] == "exact_match"),
            "wrong_number_count": sum(1 for r in rows if r["outcome"] == "wrong_number"),
            "no_prediction_count": sum(1 for r in rows if r["outcome"] == "no_prediction"),
        }

    runtime_ok = all(p.get("inference_error") is None for p in predictions)
    labels, decision = build_evidence_and_decision(
        exact, positive_accepted, negative_emission, runtime_ok
    )

    class_dist = _distribution([row["selection_class"] for row in item_rows])
    rejection_dist = _distribution([p.get("rejection_reason") for p in predictions])
    raw_nonempty = sum(1 for p in predictions if (p.get("normalized_text") or ""))
    finite_outputs = sum(
        1
        for p in predictions
        if p.get("raw_output_shape") is not None and p.get("inference_error") is None
    )

    return {
        "schema_version": RESULTS_SUMMARY_SCHEMA_VERSION,
        "status": "completed" if runtime_ok else "completed_with_inference_errors",
        "source_stage": "5C-C3D",
        "input_count": len(item_rows),
        "class_distribution": class_dist,
        "checkpoint_identity": dict(checkpoint_meta or {}),
        "positive_metrics": {
            "item_count": len(positive_rows),
            "exact_match_count": exact,
            "wrong_number_count": wrong,
            "no_prediction_count": none,
            "inference_error_count": pos_err,
            "accepted_prediction_count": positive_accepted,
            "exact_match_rate_descriptive": exact / len(positive_rows) if positive_rows else 0.0,
            "raw_non_empty_output_count": sum(
                1 for p in predictions if p["selection_class"] == POSITIVE_CLASS and (p.get("normalized_text") or "")
            ),
            "rejection_reason_distribution": _distribution(
                [p.get("rejection_reason") for p in predictions if p["selection_class"] == POSITIVE_CLASS]
            ),
            "confidence_by_outcome": {
                "exact_match": _confidence_stats(
                    [r.get("sequence_confidence") for r in positive_rows if r["outcome"] == "exact_match"]
                ),
                "wrong_number": _confidence_stats(
                    [r.get("sequence_confidence") for r in positive_rows if r["outcome"] == "wrong_number"]
                ),
                "no_prediction_rejected": _confidence_stats(
                    [
                        r.get("sequence_confidence")
                        for r in positive_rows
                        if r["outcome"] == "no_prediction"
                    ]
                ),
            },
            "one_digit_reference": _pos_subset_stats(one_digit_ref),
            "two_digit_reference": _pos_subset_stats(two_digit_ref),
            "per_manual_jersey_number": per_jersey,
        },
        "negative_metrics": {
            "item_count": len(negative_rows),
            "accepted_number_emission_count": negative_emission,
            "raw_non_empty_output_count": sum(
                1
                for p in predictions
                if p["selection_class"] != POSITIVE_CLASS and (p.get("normalized_text") or "")
            ),
            "rejected_output_count": negative_rejected,
            "emitted_number_distribution": _distribution(
                [r.get("accepted_prediction") for r in negative_rows if r["outcome"] == "number_emitted"]
            ),
            "rejection_reason_distribution": _distribution(
                [p.get("rejection_reason") for p in predictions if p["selection_class"] != POSITIVE_CLASS]
            ),
            "selection_class_distribution": _distribution(
                [r["selection_class"] for r in negative_rows]
            ),
            "confidence_distribution": _confidence_stats(
                [r.get("sequence_confidence") for r in negative_rows]
            ),
        },
        "raw_output_and_rejection": {
            "raw_non_empty_output_count_all": raw_nonempty,
            "rejection_reason_distribution_all": rejection_dist,
            "finite_output_count": finite_outputs,
            "inference_error_count": sum(1 for p in predictions if p.get("inference_error")),
            "decode_error_count": sum(
                1 for p in predictions if p.get("warning_codes") and "decode" in str(p.get("warning_codes"))
            ),
        },
        "confidence_descriptive_statistics": _confidence_stats(
            [p.get("sequence_confidence") for p in predictions]
        ),
        "evidence_labels": labels,
        "decision_class": decision,
        "comparison_with_c1_c2_descriptive_only": {
            "c1_baseline": c1_metrics or {},
            "c2_note": c2_note
            or "C2 DBNet/SAR ablation had 0/20 exact across variants; PARSeq comparison is descriptive only.",
        },
        "interpretation_limits": [
            "descriptive frozen-ROI smoke only; not deployment accuracy",
            "no confidence threshold was selected",
            "no identity/gallery/team assignment",
            "Stage 5A ROI compatibility is not a SoccerNet pose-crop claim",
            "pickled checkpoint hparams may differ from jersey decode overlay; overlay is disclosed",
        ],
        "next_gate": decision,
        "safety_flags": {
            "threshold_selected": False,
            "model_trained": False,
            "model_finetuned": False,
            "preprocessing_ablation": False,
            "dataset_downloaded": False,
            "legibility_model_used": False,
            "pose_model_used": False,
            "segment_aggregation_performed": False,
            "identity_assigned": False,
            "gallery_updated": False,
            "accuracy_claimed": False,
        },
    }


def parse_network_strace(strace_text: str) -> dict[str, Any]:
    audit = jm.parse_network_strace(strace_text)
    # C3D accepts pass_offline (no sockets) or pass_loopback_only.
    if (
        audit["external_connect_attempt_count"] == 0
        and audit["external_send_attempt_count"] == 0
        and audit["DNS_attempt_count"] == 0
        and not audit["automatic_download_attempted"]
        and audit["policy_status"] != "fail_external_network"
    ):
        if (
            audit["inet_socket_count"] == 0
            and audit["loopback_connect_count"] == 0
            and audit["loopback_bind_count"] == 0
            and not audit["loopback_socket_created"]
        ):
            audit["policy_status"] = "pass_offline"
        else:
            audit["policy_status"] = "pass_loopback_only"
    return audit


def runtime_stats(values: Sequence[Optional[float]]) -> Optional[dict[str, float]]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    ordered = sorted(finite)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(finite),
        "min_ms": min(finite),
        "max_ms": max(finite),
        "mean_ms": statistics.fmean(finite),
        "median_ms": statistics.median(finite),
        "p95_ms": ordered[p95_index],
        "sum_ms": sum(finite),
    }
