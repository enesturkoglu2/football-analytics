"""Offline CPU jersey-number OCR smoke adapter for local MMOCR DBNet+SAR assets.

Stage 5C-C1 contract highlights:
- local-path-only model init (URL and model-alias rejection);
- CPU enforcement, checkpoint/config/crop SHA-256 verification;
- Stage 5A number-search ROI reuse, no image export, no source overwrite;
- digit-only ``[0-9]{1,2}`` acceptance, no letter-to-digit conversion,
  no confidence threshold, raw model output preserved;
- manual-label blind inference with a strictly separated evaluation step;
- loopback-only network policy audit helpers (strace classification).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

INPUT_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_input_v1"
REFERENCE_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_reference_v1"
PREDICTION_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_prediction_v1"
ITEM_EVALUATION_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_item_evaluation_v1"
RESULTS_SUMMARY_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_results_summary_v1"
RUNTIME_SUMMARY_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_runtime_summary_v1"
RUN_MANIFEST_SCHEMA_VERSION = "reid_jersey_mmocr_smoke_run_manifest_v1"

DIGIT_PATTERN = re.compile(r"^[0-9]{1,2}$")
MULTI_DIGIT_PATTERN = re.compile(r"^[0-9]{3,}$")

POSITIVE_CLASS = "POS_readable"
NEGATIVE_CLASSES = (
    "A_not_visible",
    "B_visible_unreadable",
    "C_uncertain_signal",
    "D_uncertain_crop",
    "E_invalid",
)
ALL_CLASSES = (POSITIVE_CLASS,) + NEGATIVE_CLASSES

# Fields that must never leak into the blind inference manifest.
BLIND_FORBIDDEN_FIELDS = frozenset(
    {
        "manual_jersey_number",
        "manual_number_visible",
        "manual_number_readable",
        "manual_crop_valid",
        "manual_digit_count",
        "manual_back_facing",
        "manual_contamination_affects_number_region",
        "manual_notes",
        "expected_number",
        "reviewer",
        "reviewed_at",
        "identity_label",
        "team_label",
    }
)

LOOPBACK_V4_PREFIX = "127."
LOOPBACK_V6 = "::1"
WILDCARDS = ("0.0.0.0", "::")


class JerseyMMOCRError(RuntimeError):
    """Raised when the jersey MMOCR smoke contract is violated."""


# ---------------------------------------------------------------------------
# Hashing / IO helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                raise JerseyMMOCRError(f"blank line {line_no} in {path}")
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def json_safe(value: Any) -> Any:
    """Recursively convert model outputs (incl. numpy scalars/arrays) to JSON-safe values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JerseyMMOCRError(f"non-finite float in output: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):  # numpy arrays
        return json_safe(value.tolist())
    if hasattr(value, "item"):  # numpy scalars
        return json_safe(value.item())
    raise JerseyMMOCRError(f"value is not JSON-safe: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Config and local asset validation
# ---------------------------------------------------------------------------


def load_smoke_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_smoke_config(config)
    return config


def validate_smoke_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise JerseyMMOCRError("config must be a mapping")
    for key in ("schema_version", "device", "max_items", "detector", "recognizer", "digit_policy"):
        if key not in config:
            raise JerseyMMOCRError(f"config missing required key: {key}")
    if config["device"] != "cpu":
        raise JerseyMMOCRError(f"device must be 'cpu', got {config['device']!r}")
    digit_policy = config["digit_policy"]
    if digit_policy.get("confidence_threshold") is not None:
        raise JerseyMMOCRError("confidence_threshold must be null (no threshold tuning)")
    if digit_policy.get("letter_to_digit_conversion") is not False:
        raise JerseyMMOCRError("letter_to_digit_conversion must be false")
    if int(digit_policy.get("max_digits", 0)) != 2:
        raise JerseyMMOCRError("max_digits must be 2")
    for role in ("detector", "recognizer"):
        entry = config[role]
        for key in ("model_id", "config_path", "config_sha256", "checkpoint_path", "checkpoint_sha256", "checkpoint_byte_size"):
            if key not in entry:
                raise JerseyMMOCRError(f"config {role} missing key: {key}")


def validate_local_model_asset(
    path_value: str,
    expected_sha256: str,
    expected_byte_size: Optional[int] = None,
) -> Path:
    """Validate a local model config/checkpoint path. Rejects URLs and model aliases."""
    if not isinstance(path_value, str) or not path_value:
        raise JerseyMMOCRError("model asset path must be a non-empty string")
    if "://" in path_value:
        raise JerseyMMOCRError(f"URL model sources are forbidden: {path_value}")
    path = Path(path_value)
    if not path.is_absolute():
        raise JerseyMMOCRError(
            f"model must be an absolute local path (alias-like value rejected): {path_value}"
        )
    if not path.is_file():
        raise JerseyMMOCRError(f"local model asset not found: {path}")
    if expected_byte_size is not None:
        actual_size = path.stat().st_size
        if actual_size != int(expected_byte_size):
            raise JerseyMMOCRError(
                f"byte size mismatch for {path}: expected {expected_byte_size}, got {actual_size}"
            )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise JerseyMMOCRError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual_sha}"
        )
    return path


def build_inferencers(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Init TextDetInferencer/TextRecInferencer from validated local paths only (CPU)."""
    if config["device"] != "cpu":
        raise JerseyMMOCRError("only device='cpu' is allowed")
    det_cfg = validate_local_model_asset(
        config["detector"]["config_path"], config["detector"]["config_sha256"]
    )
    det_weights = validate_local_model_asset(
        config["detector"]["checkpoint_path"],
        config["detector"]["checkpoint_sha256"],
        config["detector"]["checkpoint_byte_size"],
    )
    rec_cfg = validate_local_model_asset(
        config["recognizer"]["config_path"], config["recognizer"]["config_sha256"]
    )
    rec_weights = validate_local_model_asset(
        config["recognizer"]["checkpoint_path"],
        config["recognizer"]["checkpoint_sha256"],
        config["recognizer"]["checkpoint_byte_size"],
    )

    from mmocr.apis import TextDetInferencer, TextRecInferencer

    timings: dict[str, Any] = {}
    start = time.perf_counter()
    detector = TextDetInferencer(model=str(det_cfg), weights=str(det_weights), device="cpu")
    timings["detector_init_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    recognizer = TextRecInferencer(model=str(rec_cfg), weights=str(rec_weights), device="cpu")
    timings["recognizer_init_ms"] = (time.perf_counter() - start) * 1000.0
    return detector, recognizer, timings


def build_model_meta(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detector_model_id": config["detector"]["model_id"],
        "detector_config_sha256": config["detector"]["config_sha256"],
        "detector_checkpoint_sha256": config["detector"]["checkpoint_sha256"],
        "recognizer_model_id": config["recognizer"]["model_id"],
        "recognizer_config_sha256": config["recognizer"]["config_sha256"],
        "recognizer_checkpoint_sha256": config["recognizer"]["checkpoint_sha256"],
        "device": config["device"],
        "preprocessing_variant": config.get("preprocessing_variant", "roi_bgr_no_preprocessing"),
    }


# ---------------------------------------------------------------------------
# Selection taxonomy (corrected for Stage 5C-C1)
# ---------------------------------------------------------------------------


def classify_selection_class(row: Mapping[str, Any]) -> Optional[str]:
    """Map a frozen reviewed item to a mutually exclusive selection class.

    Corrected Stage 5C-C1 taxonomy:
    - POS_readable: valid + visible=yes + readable=yes + nonblank jersey number
    - B_visible_unreadable: valid + visible=yes + readable=no (exactly; no 'uncertain')
    - A_not_visible: valid + visible=no
    - C_uncertain_signal: valid + (visible=uncertain OR readable=uncertain), no POS/A/B overlap
    - D_uncertain_crop: crop_valid=uncertain
    - E_invalid: crop_valid=invalid
    """
    crop_valid = row["manual_crop_valid"]
    visible = row["manual_number_visible"]
    readable = row["manual_number_readable"]
    jersey = (row.get("manual_jersey_number") or "").strip()
    if crop_valid == "invalid":
        return "E_invalid"
    if crop_valid == "uncertain":
        return "D_uncertain_crop"
    if crop_valid == "valid":
        if visible == "yes":
            if readable == "yes":
                if not jersey:
                    raise JerseyMMOCRError(
                        f"readable item without jersey number: {row.get('review_item_id')}"
                    )
                return POSITIVE_CLASS
            if readable == "no":
                return "B_visible_unreadable"
            if readable == "uncertain":
                return "C_uncertain_signal"
        elif visible == "no":
            return "A_not_visible"
        elif visible == "uncertain":
            return "C_uncertain_signal"
    raise JerseyMMOCRError(
        f"unclassifiable reviewed item {row.get('review_item_id')}: "
        f"crop_valid={crop_valid!r} visible={visible!r} readable={readable!r}"
    )


def build_selection(
    reviewed_rows: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[str, int],
    *,
    a_max_per_segment: int = 1,
) -> list[dict[str, Any]]:
    """Deterministically select the smoke set from the frozen reviewed items."""
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in ALL_CLASSES}
    for row in reviewed_rows:
        selection_class = classify_selection_class(row)
        if selection_class is not None:
            entry = dict(row)
            entry["selection_class"] = selection_class
            by_class[selection_class].append(entry)
    for name in ALL_CLASSES:
        by_class[name].sort(key=lambda item: int(item["pilot_index"]))

    selected: list[dict[str, Any]] = []
    # POS_readable, B and C: take the whole class population.
    for name in (POSITIVE_CLASS, "B_visible_unreadable", "C_uncertain_signal"):
        selected.extend(by_class[name])

    # A_not_visible: contamination=yes first, then pilot_index; max 1 per segment.
    a_candidates = sorted(
        by_class["A_not_visible"],
        key=lambda item: (
            0 if item.get("manual_contamination_affects_number_region") == "yes" else 1,
            int(item["pilot_index"]),
        ),
    )
    a_selected: list[dict[str, Any]] = []
    used_segments: dict[str, int] = {}
    for candidate in a_candidates:
        if len(a_selected) >= expected_counts["A_not_visible"]:
            break
        segment_id = str(candidate["segment_id"])
        if used_segments.get(segment_id, 0) >= a_max_per_segment:
            continue
        used_segments[segment_id] = used_segments.get(segment_id, 0) + 1
        a_selected.append(candidate)
    selected.extend(a_selected)

    # D and E: first-N by ascending pilot_index.
    selected.extend(by_class["D_uncertain_crop"][: expected_counts["D_uncertain_crop"]])
    selected.extend(by_class["E_invalid"][: expected_counts["E_invalid"]])

    selected.sort(key=lambda item: int(item["pilot_index"]))
    validate_selection(selected, expected_counts)
    return selected


def validate_selection(
    selected: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[str, int],
) -> None:
    counts: dict[str, int] = {}
    for item in selected:
        counts[item["selection_class"]] = counts.get(item["selection_class"], 0) + 1
    for name, expected in expected_counts.items():
        if counts.get(name, 0) != expected:
            raise JerseyMMOCRError(
                f"selection count mismatch for {name}: expected {expected}, got {counts.get(name, 0)}"
            )
    total_expected = sum(expected_counts.values())
    if len(selected) != total_expected:
        raise JerseyMMOCRError(f"selection total mismatch: expected {total_expected}, got {len(selected)}")
    review_ids = [item["review_item_id"] for item in selected]
    pilot_indices = [int(item["pilot_index"]) for item in selected]
    if len(set(review_ids)) != len(review_ids):
        raise JerseyMMOCRError("duplicate review_item_id in selection")
    if len(set(pilot_indices)) != len(pilot_indices):
        raise JerseyMMOCRError("duplicate pilot_index in selection")


# ---------------------------------------------------------------------------
# Blind inference manifest / evaluation reference
# ---------------------------------------------------------------------------


def build_blind_manifest(
    selected: Sequence[Mapping[str, Any]],
    canonical_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the blind inference manifest (no manual labels, no reviewer identity)."""
    records: list[dict[str, Any]] = []
    for item in selected:
        review_item_id = item["review_item_id"]
        canonical = canonical_by_id.get(review_item_id)
        if canonical is None:
            raise JerseyMMOCRError(f"canonical review item missing: {review_item_id}")
        if canonical["source_crop_sha256"] != item["source_crop_sha256"]:
            raise JerseyMMOCRError(f"crop sha mismatch between frozen and canonical: {review_item_id}")
        records.append(
            {
                "schema_version": INPUT_SCHEMA_VERSION,
                "pilot_index": int(item["pilot_index"]),
                "review_item_id": review_item_id,
                "crop_id": item["crop_id"],
                "segment_id": item["segment_id"],
                "raw_track_id": int(item["raw_track_id"]),
                "frame_index": int(item["frame_index"]),
                "selection_class": item["selection_class"],
                "source_crop_path": canonical["source_crop_path"],
                "source_crop_sha256": canonical["source_crop_sha256"],
                "crop_width_px": int(canonical["crop_width_px"]),
                "crop_height_px": int(canonical["crop_height_px"]),
                "roi_x_min": int(canonical["roi_x_min"]),
                "roi_y_min": int(canonical["roi_y_min"]),
                "roi_x_max": int(canonical["roi_x_max"]),
                "roi_y_max": int(canonical["roi_y_max"]),
                "roi_source": "stage5a_number_search_roi",
            }
        )
    assert_blind_records_safe(records)
    return records


def assert_blind_records_safe(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        leaked = BLIND_FORBIDDEN_FIELDS.intersection(record.keys())
        if leaked:
            raise JerseyMMOCRError(f"manual-label fields leaked into blind manifest: {sorted(leaked)}")


def build_evaluation_reference(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "pilot_index": int(item["pilot_index"]),
            "review_item_id": item["review_item_id"],
            "selection_class": item["selection_class"],
            "manual_crop_valid": item["manual_crop_valid"],
            "manual_number_visible": item["manual_number_visible"],
            "manual_number_readable": item["manual_number_readable"],
            "manual_jersey_number": item["manual_jersey_number"],
            "manual_digit_count": item["manual_digit_count"],
            "manual_contamination_affects_number_region": item[
                "manual_contamination_affects_number_region"
            ],
        }
        for item in selected
    ]


def validate_manifest_pair(
    blind_records: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
    expected_total: int,
) -> None:
    if len(blind_records) != expected_total or len(reference_records) != expected_total:
        raise JerseyMMOCRError(
            f"manifest row counts must equal {expected_total}: "
            f"blind={len(blind_records)} reference={len(reference_records)}"
        )
    blind_ids = [record["review_item_id"] for record in blind_records]
    reference_ids = [record["review_item_id"] for record in reference_records]
    if len(set(blind_ids)) != len(blind_ids) or len(set(reference_ids)) != len(reference_ids):
        raise JerseyMMOCRError("duplicate review_item_id in manifest pair")
    if blind_ids != reference_ids:
        raise JerseyMMOCRError("blind/reference manifests must join exactly in the same order")
    assert_blind_records_safe(blind_records)


# ---------------------------------------------------------------------------
# Digit policy
# ---------------------------------------------------------------------------


def normalize_recognized_text(raw_text: Optional[str]) -> str:
    """NFKC-normalize and strip raw recognizer output (no character substitution)."""
    if raw_text is None:
        return ""
    return unicodedata.normalize("NFKC", str(raw_text)).strip()


def extract_digit_candidate(normalized_text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (accepted_digit_candidate, rejection_reason) for one recognized text."""
    if not normalized_text:
        return None, "empty_text"
    if DIGIT_PATTERN.fullmatch(normalized_text):
        return normalized_text, None
    if MULTI_DIGIT_PATTERN.fullmatch(normalized_text):
        return None, "digit_count_exceeds_max"
    return None, "non_digit_text"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def clamp_roi(
    x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int
) -> Optional[tuple[int, int, int, int]]:
    x1 = max(0, min(width, int(x_min)))
    y1 = max(0, min(height, int(y_min)))
    x2 = max(0, min(width, int(x_max)))
    y2 = max(0, min(height, int(y_max)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def polygon_to_bbox(polygon: Sequence[float]) -> tuple[int, int, int, int]:
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        raise JerseyMMOCRError(f"invalid polygon length: {len(polygon)}")
    xs = [float(value) for value in polygon[0::2]]
    ys = [float(value) for value in polygon[1::2]]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


# ---------------------------------------------------------------------------
# Blind inference
# ---------------------------------------------------------------------------


def predict_single_item(
    item: Mapping[str, Any],
    detector: Any,
    recognizer: Any,
    model_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Run detector+recognizer on one blind manifest item.

    The item must come from the blind manifest (no manual labels). Crop bytes are
    SHA-256 verified before decoding; a mismatch is a hard contract failure.
    """
    import cv2

    prediction: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "pilot_index": int(item["pilot_index"]),
        "review_item_id": item["review_item_id"],
        "crop_id": item["crop_id"],
        "segment_id": item["segment_id"],
        "raw_track_id": int(item["raw_track_id"]),
        "frame_index": int(item["frame_index"]),
        "selection_class": item["selection_class"],
        "source_crop_path": item["source_crop_path"],
        "source_crop_sha256": item["source_crop_sha256"],
        "roi_source": item.get("roi_source", "stage5a_number_search_roi"),
        **dict(model_meta),
        "image_width": None,
        "image_height": None,
        "roi_xyxy": None,
        "roi_width": None,
        "roi_height": None,
        "detected_region_count": 0,
        "detected_text_regions": [],
        "raw_recognition_candidates": [],
        "selected_digit_string": None,
        "selected_recognition_confidence": None,
        "selected_detector_confidence": None,
        "selected_combined_confidence": None,
        "selected_region_index": None,
        "rejection_reason": None,
        "detector_runtime_ms": None,
        "recognizer_runtime_ms": None,
        "total_runtime_ms": None,
        "warnings": [],
        "inference_error": None,
    }

    total_start = time.perf_counter()
    crop_path = Path(item["source_crop_path"])
    actual_sha = sha256_file(crop_path)
    if actual_sha != item["source_crop_sha256"]:
        raise JerseyMMOCRError(
            f"crop sha256 mismatch for {item['review_item_id']}: "
            f"expected {item['source_crop_sha256']}, got {actual_sha}"
        )

    try:
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise JerseyMMOCRError(f"could not decode crop as BGR image: {crop_path}")
        height, width = image.shape[:2]
        prediction["image_width"] = int(width)
        prediction["image_height"] = int(height)

        roi_bounds = clamp_roi(
            item["roi_x_min"], item["roi_y_min"], item["roi_x_max"], item["roi_y_max"], width, height
        )
        if roi_bounds is None:
            prediction["rejection_reason"] = "roi_invalid"
            prediction["total_runtime_ms"] = (time.perf_counter() - total_start) * 1000.0
            return prediction
        x1, y1, x2, y2 = roi_bounds
        prediction["roi_xyxy"] = [x1, y1, x2, y2]
        prediction["roi_width"] = x2 - x1
        prediction["roi_height"] = y2 - y1
        roi_image = image[y1:y2, x1:x2]

        det_start = time.perf_counter()
        det_result = detector(roi_image, return_vis=False, progress_bar=False)
        prediction["detector_runtime_ms"] = (time.perf_counter() - det_start) * 1000.0
        det_pred = det_result["predictions"][0]
        polygons = det_pred.get("polygons", []) or []
        det_scores = det_pred.get("scores", []) or []
        prediction["detected_region_count"] = len(polygons)

        if not polygons:
            prediction["rejection_reason"] = "detector_no_region"
            prediction["total_runtime_ms"] = (time.perf_counter() - total_start) * 1000.0
            return prediction

        regions: list[dict[str, Any]] = []
        rec_total_ms = 0.0
        for region_index, polygon in enumerate(polygons):
            det_score = float(det_scores[region_index]) if region_index < len(det_scores) else None
            region: dict[str, Any] = {
                "region_index": region_index,
                "polygon": [float(value) for value in polygon],
                "detector_score": det_score,
                "region_bbox_xyxy": None,
                "raw_text": None,
                "normalized_text": None,
                "recognizer_score": None,
                "accepted_digit_candidate": None,
                "region_rejection_reason": None,
            }
            bbox = polygon_to_bbox(polygon)
            bounds = clamp_roi(bbox[0], bbox[1], bbox[2], bbox[3], roi_image.shape[1], roi_image.shape[0])
            if bounds is None:
                region["region_rejection_reason"] = "degenerate_region"
                regions.append(region)
                continue
            bx1, by1, bx2, by2 = bounds
            region["region_bbox_xyxy"] = [bx1, by1, bx2, by2]
            rec_start = time.perf_counter()
            rec_result = recognizer(
                roi_image[by1:by2, bx1:bx2], return_vis=False, progress_bar=False
            )
            rec_total_ms += (time.perf_counter() - rec_start) * 1000.0
            rec_pred = rec_result["predictions"][0]
            raw_text = rec_pred.get("text")
            rec_score = rec_pred.get("scores")
            normalized = normalize_recognized_text(raw_text)
            candidate, candidate_rejection = extract_digit_candidate(normalized)
            region["raw_text"] = raw_text
            region["normalized_text"] = normalized
            region["recognizer_score"] = float(rec_score) if rec_score is not None else None
            region["accepted_digit_candidate"] = candidate
            region["region_rejection_reason"] = candidate_rejection
            regions.append(region)
            prediction["raw_recognition_candidates"].append(
                {
                    "region_index": region_index,
                    "raw_text": raw_text,
                    "normalized_text": normalized,
                    "recognizer_score": region["recognizer_score"],
                }
            )
        prediction["recognizer_runtime_ms"] = rec_total_ms
        prediction["detected_text_regions"] = regions

        selection = select_digit_from_regions(regions)
        if selection is None:
            prediction["rejection_reason"] = "recognizer_no_digit"
        else:
            prediction["selected_digit_string"] = selection["digit_string"]
            prediction["selected_recognition_confidence"] = selection["recognizer_score"]
            prediction["selected_detector_confidence"] = selection["detector_score"]
            prediction["selected_combined_confidence"] = selection["combined_score"]
            prediction["selected_region_index"] = selection["region_index"]
    except JerseyMMOCRError:
        raise
    except Exception as error:  # pragma: no cover - depends on runtime model failures
        prediction["inference_error"] = f"{type(error).__name__}: {error}"

    prediction["total_runtime_ms"] = (time.perf_counter() - total_start) * 1000.0
    return prediction


def select_digit_from_regions(regions: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the accepted digit candidate with the highest recognizer confidence.

    No threshold is applied. Deterministic tie-break: lowest region_index wins.
    """
    best: Optional[dict[str, Any]] = None
    best_key: Optional[tuple[float, int]] = None
    for region in regions:
        candidate = region.get("accepted_digit_candidate")
        if candidate is None:
            continue
        score = region.get("recognizer_score")
        score_key = float(score) if score is not None else float("-inf")
        key = (score_key, -int(region["region_index"]))
        if best_key is None or key > best_key:
            best_key = key
            detector_score = region.get("detector_score")
            recognizer_score = region.get("recognizer_score")
            combined = (
                float(detector_score) * float(recognizer_score)
                if detector_score is not None and recognizer_score is not None
                else None
            )
            best = {
                "digit_string": candidate,
                "recognizer_score": recognizer_score,
                "detector_score": detector_score,
                "combined_score": combined,
                "region_index": int(region["region_index"]),
            }
    return best


def run_blind_inference(
    items: Sequence[Mapping[str, Any]],
    detector: Any,
    recognizer: Any,
    model_meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run blind inference over manifest items in deterministic pilot_index order."""
    assert_blind_records_safe(items)
    ordered = sorted(items, key=lambda item: int(item["pilot_index"]))
    return [predict_single_item(item, detector, recognizer, model_meta) for item in ordered]


# ---------------------------------------------------------------------------
# Evaluation (opened only after predictions are complete)
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    expected_class_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_by_id = {row["review_item_id"]: row for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise JerseyMMOCRError("duplicate review_item_id in evaluation reference")
    if len(predictions) != len(reference_rows):
        raise JerseyMMOCRError(
            f"prediction/reference row mismatch: {len(predictions)} vs {len(reference_rows)}"
        )

    item_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        review_item_id = prediction["review_item_id"]
        reference = reference_by_id.get(review_item_id)
        if reference is None:
            raise JerseyMMOCRError(f"prediction without evaluation reference: {review_item_id}")
        if reference["selection_class"] != prediction["selection_class"]:
            raise JerseyMMOCRError(f"selection_class mismatch for {review_item_id}")
        selection_class = prediction["selection_class"]
        predicted = prediction.get("selected_digit_string")
        error = prediction.get("inference_error")
        if selection_class == POSITIVE_CLASS:
            expected = (reference["manual_jersey_number"] or "").strip()
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
                "pilot_index": prediction["pilot_index"],
                "review_item_id": review_item_id,
                "selection_class": selection_class,
                "outcome": outcome,
                "predicted_digit_string": predicted,
                "manual_jersey_number": reference["manual_jersey_number"],
                "manual_crop_valid": reference["manual_crop_valid"],
                "manual_number_visible": reference["manual_number_visible"],
                "manual_number_readable": reference["manual_number_readable"],
                "selected_recognition_confidence": prediction.get("selected_recognition_confidence"),
                "selected_detector_confidence": prediction.get("selected_detector_confidence"),
                "rejection_reason": prediction.get("rejection_reason"),
                "inference_error": error,
                "detected_region_count": prediction.get("detected_region_count"),
                "total_runtime_ms": prediction.get("total_runtime_ms"),
            }
        )

    summary = build_results_summary(item_rows, predictions, expected_class_counts)
    return item_rows, summary


def _confidence_stats(values: Sequence[float]) -> Optional[dict[str, float]]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
    }


def build_results_summary(
    item_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    expected_class_counts: Mapping[str, int],
) -> dict[str, Any]:
    positive_rows = [row for row in item_rows if row["selection_class"] == POSITIVE_CLASS]
    if len(positive_rows) != expected_class_counts[POSITIVE_CLASS]:
        raise JerseyMMOCRError("positive row count mismatch in evaluation")

    per_jersey: dict[str, dict[str, Any]] = {}
    for row in positive_rows:
        jersey = row["manual_jersey_number"]
        entry = per_jersey.setdefault(
            jersey,
            {"item_count": 0, "exact_match_count": 0, "wrong_number_count": 0, "no_prediction_count": 0, "predictions": []},
        )
        entry["item_count"] += 1
        entry["predictions"].append(row["predicted_digit_string"])
        if row["outcome"] == "exact_match":
            entry["exact_match_count"] += 1
        elif row["outcome"] == "wrong_number":
            entry["wrong_number_count"] += 1
        elif row["outcome"] in ("no_prediction", "inference_error"):
            entry["no_prediction_count"] += 1

    exact_match_count = sum(1 for row in positive_rows if row["outcome"] == "exact_match")
    positive_prediction_count = sum(
        1 for row in positive_rows if row["predicted_digit_string"] is not None
    )
    positive_summary = {
        "item_count": len(positive_rows),
        "prediction_count": positive_prediction_count,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / len(positive_rows) if positive_rows else None,
        "wrong_number_count": sum(1 for row in positive_rows if row["outcome"] == "wrong_number"),
        "no_prediction_count": sum(1 for row in positive_rows if row["outcome"] == "no_prediction"),
        "rejection_count": sum(1 for row in positive_rows if row["rejection_reason"] is not None),
        "inference_error_count": sum(1 for row in positive_rows if row["outcome"] == "inference_error"),
        "per_jersey": per_jersey,
    }

    negative_classes: dict[str, dict[str, Any]] = {}
    for class_name in NEGATIVE_CLASSES:
        class_rows = [row for row in item_rows if row["selection_class"] == class_name]
        if len(class_rows) != expected_class_counts[class_name]:
            raise JerseyMMOCRError(f"negative row count mismatch for {class_name}")
        emissions = sum(1 for row in class_rows if row["outcome"] == "number_emitted")
        negative_classes[class_name] = {
            "item_count": len(class_rows),
            "number_emission_count": emissions,
            "rejection_count": sum(1 for row in class_rows if row["outcome"] == "rejected"),
            "inference_error_count": sum(1 for row in class_rows if row["outcome"] == "inference_error"),
            "emission_rate": emissions / len(class_rows) if class_rows else None,
        }

    negative_rows = [row for row in item_rows if row["selection_class"] != POSITIVE_CLASS]
    false_positive_count = sum(1 for row in negative_rows if row["outcome"] == "number_emitted")

    prediction_by_reason: dict[str, int] = {}
    for prediction in predictions:
        reason = prediction.get("rejection_reason")
        if reason is not None:
            prediction_by_reason[reason] = prediction_by_reason.get(reason, 0) + 1

    return {
        "schema_version": RESULTS_SUMMARY_SCHEMA_VERSION,
        "total_items": len(item_rows),
        "expected_class_counts": dict(expected_class_counts),
        "positive": positive_summary,
        "negative_classes": negative_classes,
        "negative_total": {
            "item_count": len(negative_rows),
            "false_positive_number_count": false_positive_count,
            "false_positive_number_rate": (
                false_positive_count / len(negative_rows) if negative_rows else None
            ),
            "invalid_crop_number_emission_count": negative_classes["E_invalid"]["number_emission_count"],
            "uncertain_crop_number_emission_count": negative_classes["D_uncertain_crop"][
                "number_emission_count"
            ],
            "visible_unreadable_number_emission_count": negative_classes["B_visible_unreadable"][
                "number_emission_count"
            ],
        },
        "counters": {
            "prediction_count": sum(
                1 for prediction in predictions if prediction.get("selected_digit_string") is not None
            ),
            "rejection_count": sum(
                1 for prediction in predictions if prediction.get("rejection_reason") is not None
            ),
            "inference_error_count": sum(
                1 for prediction in predictions if prediction.get("inference_error") is not None
            ),
            "detector_no_region_count": prediction_by_reason.get("detector_no_region", 0),
            "recognizer_no_digit_count": prediction_by_reason.get("recognizer_no_digit", 0),
            "roi_invalid_count": prediction_by_reason.get("roi_invalid", 0),
        },
        "confidence": {
            "selected_recognition": _confidence_stats(
                [prediction.get("selected_recognition_confidence") for prediction in predictions]
            ),
            "selected_detector": _confidence_stats(
                [prediction.get("selected_detector_confidence") for prediction in predictions]
            ),
        },
        "interpretation_limits": [
            "negative-class number emission is not by itself a confirmed identity error",
            "readable reference labels are crop-level human observations",
            "no confidence threshold was selected or tuned in this smoke run",
            "no segment-level aggregation was performed",
            "small smoke set results are not a general accuracy benchmark",
        ],
    }


# ---------------------------------------------------------------------------
# Network audit (loopback-only policy, Stage 5C-C1)
# ---------------------------------------------------------------------------

_ADDR_V4_RE = re.compile(r'inet_addr\("([^"]+)"\)')
_ADDR_V6_RE = re.compile(r'inet_pton\(AF_INET6,\s*"([^"]+)"')
_PORT_RE = re.compile(r"sin6?_port=htons\((\d+)\)")


def _classify_address(address: Optional[str]) -> str:
    if address is None:
        return "unknown"
    if address in WILDCARDS:
        return "wildcard"
    if address.startswith(LOOPBACK_V4_PREFIX) or address == LOOPBACK_V6:
        return "loopback"
    if address.startswith("::ffff:") and address[len("::ffff:") :].startswith(LOOPBACK_V4_PREFIX):
        return "loopback"
    return "external"


def parse_network_strace(strace_text: str) -> dict[str, Any]:
    """Classify strace network syscalls under the Stage 5C-C1 loopback-only policy."""
    audit = {
        "unshare_supported": False,
        "strace_used": True,
        "loopback_socket_created": False,
        "inet_socket_count": 0,
        "unix_socket_count": 0,
        "loopback_bind_count": 0,
        "loopback_connect_count": 0,
        "loopback_listen_count": 0,
        "external_connect_attempt_count": 0,
        "external_send_attempt_count": 0,
        "wildcard_bind_count": 0,
        "DNS_attempt_count": 0,
        "automatic_download_attempted": False,
        "violations": [],
        "policy_status": None,
    }
    for line in strace_text.splitlines():
        if " resumed" in line:
            continue
        v4 = _ADDR_V4_RE.search(line)
        v6 = _ADDR_V6_RE.search(line)
        address = v4.group(1) if v4 else (v6.group(1) if v6 else None)
        port_match = _PORT_RE.search(line)
        port = int(port_match.group(1)) if port_match else None
        kind = _classify_address(address)

        if re.search(r"\bsocket\(", line):
            if "AF_UNIX" in line:
                audit["unix_socket_count"] += 1
            elif "AF_INET" in line:  # matches AF_INET and AF_INET6
                audit["inet_socket_count"] += 1
                audit["loopback_socket_created"] = True
            continue
        if address is None:
            continue
        if port == 53:
            audit["DNS_attempt_count"] += 1
            audit["violations"].append(line.strip())
        if re.search(r"\bbind\(", line):
            if kind == "loopback":
                audit["loopback_bind_count"] += 1
            elif kind == "wildcard":
                audit["wildcard_bind_count"] += 1
                audit["violations"].append(line.strip())
            elif kind == "external":
                audit["violations"].append(line.strip())
        elif re.search(r"\bconnect\(", line):
            if kind == "loopback":
                audit["loopback_connect_count"] += 1
            else:
                audit["external_connect_attempt_count"] += 1
                audit["violations"].append(line.strip())
        elif re.search(r"\b(sendto|sendmsg)\(", line):
            if kind not in ("loopback", "unknown"):
                audit["external_send_attempt_count"] += 1
                audit["violations"].append(line.strip())
        elif re.search(r"\blisten\(", line):
            audit["loopback_listen_count"] += 1

    passed = (
        audit["external_connect_attempt_count"] == 0
        and audit["external_send_attempt_count"] == 0
        and audit["wildcard_bind_count"] == 0
        and audit["DNS_attempt_count"] == 0
        and not audit["automatic_download_attempted"]
    )
    audit["policy_status"] = "pass_loopback_only" if passed else "fail_external_network"
    return audit


# ---------------------------------------------------------------------------
# Runtime summary
# ---------------------------------------------------------------------------


def build_runtime_summary(
    predictions: Sequence[Mapping[str, Any]],
    init_timings: Mapping[str, float],
    environment: Mapping[str, Any],
    peak_rss_kb: Optional[int],
    warning_summary: Mapping[str, Any],
) -> dict[str, Any]:
    def stats(values: Sequence[Optional[float]]) -> Optional[dict[str, float]]:
        finite = [float(value) for value in values if value is not None]
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

    return {
        "schema_version": RUNTIME_SUMMARY_SCHEMA_VERSION,
        "detector_init_ms": init_timings.get("detector_init_ms"),
        "recognizer_init_ms": init_timings.get("recognizer_init_ms"),
        "per_item_total": stats([prediction.get("total_runtime_ms") for prediction in predictions]),
        "per_item_detector": stats(
            [prediction.get("detector_runtime_ms") for prediction in predictions]
        ),
        "per_item_recognizer": stats(
            [prediction.get("recognizer_runtime_ms") for prediction in predictions]
        ),
        "peak_rss_kb": peak_rss_kb,
        "environment": dict(environment),
        "warnings": dict(warning_summary),
        "input_color_convention": "bgr",
    }


# ---------------------------------------------------------------------------
# Stage 5C-C2 controlled recognizer/preprocessing ablation
# ---------------------------------------------------------------------------
# The C1 baseline contract above is unchanged. The additions below implement
# a fixed four-variant experiment matrix (direct SAR at 1x/2x/4x and
# DBNet+SAR at 4x) with deterministic INTER_CUBIC upscaling only. No other
# preprocessing (sharpening, contrast, CLAHE, denoise, rotation, padding,
# channel conversion) is available through these helpers.

ABLATION_PREDICTION_SCHEMA_VERSION = "reid_jersey_mmocr_ablation_prediction_v1"
ABLATION_ITEM_EVALUATION_SCHEMA_VERSION = "reid_jersey_mmocr_ablation_item_evaluation_v1"
ABLATION_VARIANT_SUMMARY_SCHEMA_VERSION = "reid_jersey_mmocr_ablation_variant_summary_v1"
ABLATION_COMPARISON_SCHEMA_VERSION = "reid_jersey_mmocr_ablation_comparison_summary_v1"
ABLATION_RUNTIME_SCHEMA_VERSION = "reid_jersey_mmocr_ablation_runtime_summary_v1"
ABLATION_RUN_MANIFEST_SCHEMA_VERSION = "reid_jersey_mmocr_ablation_run_manifest_v1"

ALLOWED_SCALE_FACTORS = (1, 2, 4)
ABLATION_INTERPOLATION = "INTER_CUBIC"

# Fixed experiment matrix. Order is part of the contract and must not change.
ABLATION_VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant_id": "direct_sar_roi_1x", "detector_used": False, "scale_factor": 1},
    {"variant_id": "direct_sar_roi_2x_cubic", "detector_used": False, "scale_factor": 2},
    {"variant_id": "direct_sar_roi_4x_cubic", "detector_used": False, "scale_factor": 4},
    {"variant_id": "dbnet_sar_roi_4x_cubic", "detector_used": True, "scale_factor": 4},
)
ABLATION_VARIANT_IDS = tuple(variant["variant_id"] for variant in ABLATION_VARIANTS)

# Evidence labels (descriptive only; no deployment decision).
LABEL_DIRECT_EXACT = "DIRECT_RECOGNIZER_EXACT_SIGNAL_PRESENT"
LABEL_UPSCALE_RECOGNITION = "UPSCALE_IMPROVES_DIRECT_RECOGNITION"
LABEL_UPSCALE_DETECTION = "UPSCALE_IMPROVES_DETECTION"
LABEL_NO_EXACT = "NO_EXACT_SIGNAL_IN_TESTED_VARIANTS"
LABEL_NEGATIVE_RISK = "NEGATIVE_EMISSION_RISK_PRESENT"


def resize_roi_deterministic(roi_image: Any, scale_factor: int) -> tuple[Any, float]:
    """Deterministically upscale a BGR ROI by an exact integer factor.

    Only ``cv2.INTER_CUBIC`` is used; aspect ratio is preserved by scaling
    both dimensions with the same factor. ``scale_factor == 1`` returns the
    input array unchanged (no resize call, no copy semantics change).
    Returns ``(image, resize_runtime_ms)``.
    """
    import cv2

    if scale_factor not in ALLOWED_SCALE_FACTORS:
        raise JerseyMMOCRError(f"scale_factor must be one of {ALLOWED_SCALE_FACTORS}: {scale_factor}")
    if roi_image is None or getattr(roi_image, "size", 0) == 0:
        raise JerseyMMOCRError("cannot resize an empty ROI")
    if roi_image.ndim != 3 or roi_image.shape[2] != 3:
        raise JerseyMMOCRError("ROI must be an HxWx3 BGR array")
    if scale_factor == 1:
        return roi_image, 0.0
    height, width = roi_image.shape[:2]
    start = time.perf_counter()
    resized = cv2.resize(
        roi_image,
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_CUBIC,
    )
    resize_ms = (time.perf_counter() - start) * 1000.0
    if resized.shape[:2] != (height * scale_factor, width * scale_factor):
        raise JerseyMMOCRError(
            f"invalid resize result: expected {(height * scale_factor, width * scale_factor)}, "
            f"got {resized.shape[:2]}"
        )
    return resized, resize_ms


def scale_coords_to_original(values: Sequence[float], scale_factor: int) -> list[float]:
    """Map scaled-ROI coordinates back to the original ROI coordinate system."""
    if scale_factor not in ALLOWED_SCALE_FACTORS:
        raise JerseyMMOCRError(f"scale_factor must be one of {ALLOWED_SCALE_FACTORS}: {scale_factor}")
    return [float(value) / float(scale_factor) for value in values]


def extract_recognition_candidates(rec_pred: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize a recognizer prediction into an ordered candidate list.

    MMOCR's ``TextRecInferencer`` normally returns a single ``text``/``scores``
    pair; if a list is returned, the raw API order is preserved.
    """
    raw_text = rec_pred.get("text")
    raw_score = rec_pred.get("scores")
    if isinstance(raw_text, (list, tuple)):
        texts = list(raw_text)
        scores = list(raw_score) if isinstance(raw_score, (list, tuple)) else [raw_score] * len(texts)
    else:
        texts = [raw_text]
        scores = [raw_score]
    candidates: list[dict[str, Any]] = []
    for api_index, (text, score) in enumerate(zip(texts, scores)):
        normalized = normalize_recognized_text(text)
        digit, rejection = extract_digit_candidate(normalized)
        candidates.append(
            {
                "api_index": api_index,
                "raw_text": text,
                "normalized_text": normalized,
                "recognizer_score": float(score) if score is not None else None,
                "accepted_digit_candidate": digit,
                "candidate_rejection_reason": rejection,
            }
        )
    return candidates


def select_direct_sar_digit(candidates: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Select the strict digit candidate for a direct SAR call.

    Rules (no threshold): highest recognizer confidence wins; equal
    confidence resolves by API order; if no candidate carries confidence,
    the deterministic first strict digit candidate is selected.
    """
    strict = [c for c in candidates if c.get("accepted_digit_candidate") is not None]
    if not strict:
        return None
    if all(c.get("recognizer_score") is None for c in strict):
        best = strict[0]
    else:
        best = max(
            strict,
            key=lambda c: (
                c["recognizer_score"] if c.get("recognizer_score") is not None else float("-inf"),
                -int(c["api_index"]),
            ),
        )
    return {
        "digit_string": best["accepted_digit_candidate"],
        "recognizer_score": best.get("recognizer_score"),
        "api_index": int(best["api_index"]),
    }


def validate_ablation_variants(variants: Sequence[Mapping[str, Any]]) -> None:
    """Enforce the fixed Stage 5C-C2 experiment matrix (order included)."""
    got = [
        (v["variant_id"], bool(v["detector_used"]), int(v["scale_factor"]))
        for v in variants
    ]
    expected = [
        (v["variant_id"], bool(v["detector_used"]), int(v["scale_factor"]))
        for v in ABLATION_VARIANTS
    ]
    if got != expected:
        raise JerseyMMOCRError(f"ablation variant matrix mismatch: {got} != {expected}")


def predict_item_ablation_variant(
    item: Mapping[str, Any],
    variant: Mapping[str, Any],
    detector: Any,
    recognizer: Any,
    model_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one blind-manifest item through one ablation variant.

    ROI coordinates come unchanged from the Stage 5A number-search ROI in the
    blind manifest; the only allowed transform is the exact integer
    INTER_CUBIC upscale defined by the variant.
    """
    import cv2

    variant_id = variant["variant_id"]
    detector_used = bool(variant["detector_used"])
    scale_factor = int(variant["scale_factor"])
    prediction: dict[str, Any] = {
        "schema_version": ABLATION_PREDICTION_SCHEMA_VERSION,
        "review_item_id": item["review_item_id"],
        "pilot_index": int(item["pilot_index"]),
        "selection_class": item["selection_class"],
        "crop_id": item["crop_id"],
        "segment_id": item["segment_id"],
        "raw_track_id": int(item["raw_track_id"]),
        "frame_index": int(item["frame_index"]),
        "source_crop_path": item["source_crop_path"],
        "source_crop_sha256": item["source_crop_sha256"],
        **dict(model_meta),
        "variant_id": variant_id,
        "detector_used": detector_used,
        "recognizer_used": True,
        "recognition_scope": (
            "detector_regions" if detector_used else "full_number_search_roi"
        ),
        "scale_factor": scale_factor,
        "interpolation": ABLATION_INTERPOLATION if scale_factor != 1 else None,
        "preprocessing_variant": f"number_search_roi_{variant_id}",
        "color_convention": "bgr",
        "original_image_width": None,
        "original_image_height": None,
        "roi_xyxy": None,
        "original_roi_width": None,
        "original_roi_height": None,
        "processed_roi_width": None,
        "processed_roi_height": None,
        "detected_text_regions": [],
        "detector_region_count": None,
        "recognizer_call_count": 0,
        "raw_recognition_candidates": [],
        "selected_digit_string": None,
        "selected_recognition_confidence": None,
        "selected_detector_confidence": None,
        "selected_combined_confidence": None,
        "rejection_reason": None,
        "resize_runtime_ms": None,
        "detector_runtime_ms": None,
        "recognizer_runtime_ms": None,
        "total_runtime_ms": None,
        "warnings": [],
        "inference_error": None,
    }

    total_start = time.perf_counter()
    crop_path = Path(item["source_crop_path"])
    actual_sha = sha256_file(crop_path)
    if actual_sha != item["source_crop_sha256"]:
        raise JerseyMMOCRError(
            f"crop sha256 mismatch for {item['review_item_id']}: "
            f"expected {item['source_crop_sha256']}, got {actual_sha}"
        )

    try:
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise JerseyMMOCRError(f"could not decode crop as BGR image: {crop_path}")
        height, width = image.shape[:2]
        prediction["original_image_width"] = int(width)
        prediction["original_image_height"] = int(height)

        roi_bounds = clamp_roi(
            item["roi_x_min"], item["roi_y_min"], item["roi_x_max"], item["roi_y_max"], width, height
        )
        if roi_bounds is None:
            prediction["rejection_reason"] = "roi_invalid"
            prediction["total_runtime_ms"] = (time.perf_counter() - total_start) * 1000.0
            return prediction
        x1, y1, x2, y2 = roi_bounds
        prediction["roi_xyxy"] = [x1, y1, x2, y2]
        prediction["original_roi_width"] = x2 - x1
        prediction["original_roi_height"] = y2 - y1
        roi_image = image[y1:y2, x1:x2]

        processed, resize_ms = resize_roi_deterministic(roi_image, scale_factor)
        prediction["resize_runtime_ms"] = resize_ms
        prediction["processed_roi_width"] = int(processed.shape[1])
        prediction["processed_roi_height"] = int(processed.shape[0])

        if not detector_used:
            rec_start = time.perf_counter()
            rec_result = recognizer(processed, return_vis=False, progress_bar=False)
            prediction["recognizer_runtime_ms"] = (time.perf_counter() - rec_start) * 1000.0
            prediction["recognizer_call_count"] = 1
            rec_pred = rec_result["predictions"][0]
            candidates = extract_recognition_candidates(rec_pred)
            prediction["raw_recognition_candidates"] = candidates
            selection = select_direct_sar_digit(candidates)
            if selection is None:
                reasons = [c["candidate_rejection_reason"] for c in candidates]
                prediction["rejection_reason"] = reasons[0] if reasons else "recognizer_no_digit"
            else:
                prediction["selected_digit_string"] = selection["digit_string"]
                prediction["selected_recognition_confidence"] = selection["recognizer_score"]
                prediction["selected_combined_confidence"] = selection["recognizer_score"]
        else:
            det_start = time.perf_counter()
            det_result = detector(processed, return_vis=False, progress_bar=False)
            prediction["detector_runtime_ms"] = (time.perf_counter() - det_start) * 1000.0
            det_pred = det_result["predictions"][0]
            polygons = det_pred.get("polygons", []) or []
            det_scores = det_pred.get("scores", []) or []
            prediction["detector_region_count"] = len(polygons)
            if not polygons:
                prediction["rejection_reason"] = "detector_no_region"
                prediction["total_runtime_ms"] = (time.perf_counter() - total_start) * 1000.0
                return prediction
            regions: list[dict[str, Any]] = []
            rec_total_ms = 0.0
            for region_index, polygon in enumerate(polygons):
                det_score = (
                    float(det_scores[region_index]) if region_index < len(det_scores) else None
                )
                region: dict[str, Any] = {
                    "region_index": region_index,
                    "polygon_scaled": [float(v) for v in polygon],
                    "polygon_original": scale_coords_to_original(polygon, scale_factor),
                    "detector_score": det_score,
                    "region_bbox_scaled_xyxy": None,
                    "region_bbox_original_xyxy": None,
                    "raw_text": None,
                    "normalized_text": None,
                    "recognizer_score": None,
                    "accepted_digit_candidate": None,
                    "region_rejection_reason": None,
                }
                bbox = polygon_to_bbox(polygon)
                bounds = clamp_roi(
                    bbox[0], bbox[1], bbox[2], bbox[3], processed.shape[1], processed.shape[0]
                )
                if bounds is None:
                    region["region_rejection_reason"] = "degenerate_region"
                    regions.append(region)
                    continue
                bx1, by1, bx2, by2 = bounds
                region["region_bbox_scaled_xyxy"] = [bx1, by1, bx2, by2]
                region["region_bbox_original_xyxy"] = scale_coords_to_original(
                    [bx1, by1, bx2, by2], scale_factor
                )
                rec_start = time.perf_counter()
                rec_result = recognizer(
                    processed[by1:by2, bx1:bx2], return_vis=False, progress_bar=False
                )
                rec_total_ms += (time.perf_counter() - rec_start) * 1000.0
                prediction["recognizer_call_count"] += 1
                rec_pred = rec_result["predictions"][0]
                raw_text = rec_pred.get("text")
                rec_score = rec_pred.get("scores")
                normalized = normalize_recognized_text(raw_text)
                digit, rejection = extract_digit_candidate(normalized)
                region["raw_text"] = raw_text
                region["normalized_text"] = normalized
                region["recognizer_score"] = float(rec_score) if rec_score is not None else None
                region["accepted_digit_candidate"] = digit
                region["region_rejection_reason"] = rejection
                regions.append(region)
                prediction["raw_recognition_candidates"].append(
                    {
                        "region_index": region_index,
                        "raw_text": raw_text,
                        "normalized_text": normalized,
                        "recognizer_score": region["recognizer_score"],
                    }
                )
            prediction["recognizer_runtime_ms"] = rec_total_ms
            prediction["detected_text_regions"] = regions
            selection = select_digit_from_regions(regions)
            if selection is None:
                prediction["rejection_reason"] = "recognizer_no_digit"
            else:
                prediction["selected_digit_string"] = selection["digit_string"]
                prediction["selected_recognition_confidence"] = selection["recognizer_score"]
                prediction["selected_detector_confidence"] = selection["detector_score"]
                prediction["selected_combined_confidence"] = selection["combined_score"]
    except JerseyMMOCRError:
        raise
    except Exception as error:  # pragma: no cover - depends on runtime model failures
        prediction["inference_error"] = f"{type(error).__name__}: {error}"

    prediction["total_runtime_ms"] = (time.perf_counter() - total_start) * 1000.0
    return prediction


def run_ablation_inference(
    items: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    detector: Any,
    recognizer: Any,
    model_meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run the fixed variant matrix over the blind manifest.

    Output ordering is variant-major (exact contract order) with
    pilot_index ascending inside each variant: 46 items x 4 variants = 184.
    """
    validate_ablation_variants(variants)
    assert_blind_records_safe(items)
    ordered_items = sorted(items, key=lambda item: int(item["pilot_index"]))
    predictions: list[dict[str, Any]] = []
    for variant in variants:
        for item in ordered_items:
            predictions.append(
                predict_item_ablation_variant(item, variant, detector, recognizer, model_meta)
            )
    expected_total = len(ordered_items) * len(variants)
    if len(predictions) != expected_total:
        raise JerseyMMOCRError(
            f"ablation prediction count mismatch: expected {expected_total}, got {len(predictions)}"
        )
    return predictions


def _ablation_outcome(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    predicted = prediction.get("selected_digit_string")
    if prediction.get("inference_error") is not None:
        return "inference_error"
    if prediction["selection_class"] == POSITIVE_CLASS:
        expected = (reference["manual_jersey_number"] or "").strip()
        if predicted is None:
            return "no_prediction"
        return "exact_match" if predicted == expected else "wrong_number"
    return "number_emitted" if predicted is not None else "rejected"


def evaluate_ablation_predictions(
    predictions: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    expected_class_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join 184 variant predictions with the 46-row evaluation reference."""
    reference_by_id = {row["review_item_id"]: row for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise JerseyMMOCRError("duplicate review_item_id in ablation evaluation reference")
    expected_total = len(reference_rows) * len(ABLATION_VARIANT_IDS)
    if len(predictions) != expected_total:
        raise JerseyMMOCRError(
            f"ablation join mismatch: {len(predictions)} predictions != {expected_total}"
        )

    item_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        reference = reference_by_id.get(prediction["review_item_id"])
        if reference is None:
            raise JerseyMMOCRError(
                f"ablation prediction without reference: {prediction['review_item_id']}"
            )
        if reference["selection_class"] != prediction["selection_class"]:
            raise JerseyMMOCRError(
                f"selection_class mismatch for {prediction['review_item_id']}"
            )
        outcome = _ablation_outcome(prediction, reference)
        item_rows.append(
            {
                "schema_version": ABLATION_ITEM_EVALUATION_SCHEMA_VERSION,
                "variant_id": prediction["variant_id"],
                "pilot_index": prediction["pilot_index"],
                "review_item_id": prediction["review_item_id"],
                "selection_class": prediction["selection_class"],
                "outcome": outcome,
                "predicted_digit_string": prediction.get("selected_digit_string"),
                "manual_jersey_number": reference["manual_jersey_number"],
                "manual_crop_valid": reference["manual_crop_valid"],
                "manual_number_visible": reference["manual_number_visible"],
                "manual_number_readable": reference["manual_number_readable"],
                "selected_recognition_confidence": prediction.get(
                    "selected_recognition_confidence"
                ),
                "selected_detector_confidence": prediction.get("selected_detector_confidence"),
                "rejection_reason": prediction.get("rejection_reason"),
                "inference_error": prediction.get("inference_error"),
                "detector_region_count": prediction.get("detector_region_count"),
                "total_runtime_ms": prediction.get("total_runtime_ms"),
            }
        )
    variant_summary = build_ablation_variant_summary(
        item_rows, predictions, expected_class_counts
    )
    return item_rows, variant_summary


def _distribution(values: Sequence[Optional[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value if value is not None else "null"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_ablation_variant_summary(
    item_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    expected_class_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Per-variant positive/negative metrics for the fixed matrix."""
    per_variant: dict[str, Any] = {}
    for variant_id in ABLATION_VARIANT_IDS:
        rows = [row for row in item_rows if row["variant_id"] == variant_id]
        preds = [p for p in predictions if p["variant_id"] == variant_id]
        if len(rows) != sum(expected_class_counts.values()):
            raise JerseyMMOCRError(f"variant row count mismatch for {variant_id}")
        positive = [row for row in rows if row["selection_class"] == POSITIVE_CLASS]
        negative = [row for row in rows if row["selection_class"] != POSITIVE_CLASS]
        positive_preds = [p for p in preds if p["selection_class"] == POSITIVE_CLASS]
        exact = sum(1 for row in positive if row["outcome"] == "exact_match")
        per_jersey: dict[str, dict[str, int]] = {}
        for row in positive:
            entry = per_jersey.setdefault(
                row["manual_jersey_number"], {"item_count": 0, "exact_match_count": 0}
            )
            entry["item_count"] += 1
            if row["outcome"] == "exact_match":
                entry["exact_match_count"] += 1
        raw_texts: list[Optional[str]] = []
        for p in positive_preds:
            for candidate in p.get("raw_recognition_candidates", []):
                raw_texts.append(candidate.get("normalized_text"))
        direct_non_digit = 0
        direct_empty = 0
        for p in preds:
            if p.get("detector_used"):
                continue
            for candidate in p.get("raw_recognition_candidates", []):
                if candidate.get("candidate_rejection_reason") == "empty_text":
                    direct_empty += 1
                elif candidate.get("candidate_rejection_reason") is not None:
                    direct_non_digit += 1
        region_counts = [
            p.get("detector_region_count")
            for p in preds
            if p.get("detector_used")
        ]
        negative_classes: dict[str, dict[str, Any]] = {}
        for class_name in NEGATIVE_CLASSES:
            class_rows = [row for row in negative if row["selection_class"] == class_name]
            emissions = sum(1 for row in class_rows if row["outcome"] == "number_emitted")
            negative_classes[class_name] = {
                "item_count": len(class_rows),
                "number_emission_count": emissions,
                "rejection_count": sum(1 for row in class_rows if row["outcome"] == "rejected"),
                "inference_error_count": sum(
                    1 for row in class_rows if row["outcome"] == "inference_error"
                ),
                "emission_rate": emissions / len(class_rows) if class_rows else None,
            }
        runtimes = [p.get("total_runtime_ms") for p in preds if p.get("total_runtime_ms") is not None]
        ordered_runtimes = sorted(float(v) for v in runtimes)
        p95_index = max(0, math.ceil(0.95 * len(ordered_runtimes)) - 1) if ordered_runtimes else None
        per_variant[variant_id] = {
            "positive": {
                "item_count": len(positive),
                "exact_match_count": exact,
                "exact_match_rate": exact / len(positive) if positive else None,
                "number_emission_count": sum(
                    1 for row in positive if row["predicted_digit_string"] is not None
                ),
                "no_prediction_count": sum(1 for row in positive if row["outcome"] == "no_prediction"),
                "wrong_number_count": sum(1 for row in positive if row["outcome"] == "wrong_number"),
                "inference_error_count": sum(
                    1 for row in positive if row["outcome"] == "inference_error"
                ),
                "per_jersey_exact_match": per_jersey,
                "raw_text_distribution": _distribution(raw_texts),
                "rejection_reason_distribution": _distribution(
                    [row["rejection_reason"] for row in positive]
                ),
                "recognition_confidence": _confidence_stats(
                    [p.get("selected_recognition_confidence") for p in positive_preds]
                ),
            },
            "negative": {
                "item_count": len(negative),
                "number_emission_count": sum(
                    1 for row in negative if row["outcome"] == "number_emitted"
                ),
                "emission_rate": (
                    sum(1 for row in negative if row["outcome"] == "number_emitted") / len(negative)
                    if negative
                    else None
                ),
                "rejection_count": sum(1 for row in negative if row["outcome"] == "rejected"),
                "inference_error_count": sum(
                    1 for row in negative if row["outcome"] == "inference_error"
                ),
                "per_class": negative_classes,
                "invalid_crop_number_emission_count": negative_classes["E_invalid"][
                    "number_emission_count"
                ],
                "uncertain_signal_number_emission_count": negative_classes["C_uncertain_signal"][
                    "number_emission_count"
                ],
                "visible_unreadable_number_emission_count": negative_classes[
                    "B_visible_unreadable"
                ]["number_emission_count"],
            },
            "counters": {
                "direct_sar_non_digit_output_count": direct_non_digit,
                "direct_sar_empty_output_count": direct_empty,
                "detector_region_item_count": sum(
                    1 for count in region_counts if count is not None and count > 0
                ),
                "detector_no_region_item_count": sum(
                    1 for count in region_counts if count == 0
                ),
                "detector_total_region_count": sum(
                    int(count) for count in region_counts if count is not None
                ),
                "inference_error_count": sum(
                    1 for p in preds if p.get("inference_error") is not None
                ),
            },
            "runtime": {
                "mean_ms": statistics.fmean(ordered_runtimes) if ordered_runtimes else None,
                "median_ms": statistics.median(ordered_runtimes) if ordered_runtimes else None,
                "p95_ms": ordered_runtimes[p95_index] if ordered_runtimes else None,
                "sum_ms": sum(ordered_runtimes) if ordered_runtimes else None,
            },
        }
    return {
        "schema_version": ABLATION_VARIANT_SUMMARY_SCHEMA_VERSION,
        "variant_order": list(ABLATION_VARIANT_IDS),
        "expected_class_counts": dict(expected_class_counts),
        "per_variant": per_variant,
    }


def build_ablation_comparison_summary(
    variant_summary: Mapping[str, Any],
    c1_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic descriptive comparison against the frozen C1 baseline.

    Ranking is descriptive only and is not an automatic deployment decision.
    """
    per_variant = variant_summary["per_variant"]

    def row(variant_id: str) -> dict[str, Any]:
        entry = per_variant[variant_id]
        return {
            "variant_id": variant_id,
            "positive_exact_match_count": entry["positive"]["exact_match_count"],
            "positive_number_emission_count": entry["positive"]["number_emission_count"],
            "positive_wrong_number_count": entry["positive"]["wrong_number_count"],
            "positive_no_prediction_count": entry["positive"]["no_prediction_count"],
            "negative_number_emission_count": entry["negative"]["number_emission_count"],
            "median_runtime_ms": entry["runtime"]["median_ms"],
        }

    table = [
        {
            "variant_id": "c1_dbnet_sar_roi_1x_baseline",
            "positive_exact_match_count": c1_baseline["positive_exact_match_count"],
            "positive_number_emission_count": c1_baseline["positive_number_emission_count"],
            "positive_wrong_number_count": c1_baseline["positive_wrong_number_count"],
            "positive_no_prediction_count": c1_baseline["positive_no_prediction_count"],
            "negative_number_emission_count": c1_baseline["negative_number_emission_count"],
            "median_runtime_ms": c1_baseline["median_runtime_ms"],
            "is_frozen_baseline": True,
        }
    ] + [row(variant_id) for variant_id in ABLATION_VARIANT_IDS]

    def ranking_key(entry: Mapping[str, Any]) -> tuple:
        return (
            -entry["positive_exact_match_count"],
            entry["negative_number_emission_count"],
            entry["positive_wrong_number_count"],
            entry["positive_no_prediction_count"],
            entry["median_runtime_ms"] if entry["median_runtime_ms"] is not None else float("inf"),
            ABLATION_VARIANT_IDS.index(entry["variant_id"]),
        )

    ranked = sorted(
        (row(variant_id) for variant_id in ABLATION_VARIANT_IDS), key=ranking_key
    )

    direct_ids = [v for v in ABLATION_VARIANT_IDS if v.startswith("direct_sar")]
    direct_exact = {v: per_variant[v]["positive"]["exact_match_count"] for v in direct_ids}
    labels: list[str] = []
    if any(count > 0 for count in direct_exact.values()):
        labels.append(LABEL_DIRECT_EXACT)
    if (
        direct_exact["direct_sar_roi_2x_cubic"] > direct_exact["direct_sar_roi_1x"]
        or direct_exact["direct_sar_roi_4x_cubic"] > direct_exact["direct_sar_roi_1x"]
    ):
        labels.append(LABEL_UPSCALE_RECOGNITION)
    if (
        per_variant["dbnet_sar_roi_4x_cubic"]["counters"]["detector_region_item_count"]
        > c1_baseline["detector_region_item_count"]
    ):
        labels.append(LABEL_UPSCALE_DETECTION)
    if all(
        per_variant[v]["positive"]["exact_match_count"] == 0 for v in ABLATION_VARIANT_IDS
    ):
        labels.append(LABEL_NO_EXACT)
    if any(
        per_variant[v]["negative"]["number_emission_count"] > 0 for v in ABLATION_VARIANT_IDS
    ):
        labels.append(LABEL_NEGATIVE_RISK)

    return {
        "schema_version": ABLATION_COMPARISON_SCHEMA_VERSION,
        "c1_baseline": dict(c1_baseline),
        "comparison_table": table,
        "descriptive_ranking": [entry["variant_id"] for entry in ranked],
        "ranking_criteria": [
            "higher positive exact_match_count",
            "lower negative number_emission_count",
            "lower positive wrong_number_count",
            "lower positive no_prediction_count",
            "lower median runtime",
        ],
        "ranking_is_deployment_decision": False,
        "evidence_labels": labels,
        "confidence_note": (
            "Recognizer confidences are not calibrated between the detector "
            "pipeline and the direct-recognizer pipeline; values are not "
            "directly comparable across variants."
        ),
        "interpretation_limits": [
            "46 deterministic pilot crops only; not a general accuracy benchmark",
            "no threshold selection or confidence calibration was performed",
            "only exact integer INTER_CUBIC upscaling was tested; no other preprocessing",
            "ROI coordinates, source crops, checkpoints, and configs are unchanged from C1",
            "no segment aggregation and no identity decision were made",
        ],
    }
