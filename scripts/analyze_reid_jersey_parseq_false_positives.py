#!/usr/bin/env python
"""Stage 5C-C3E read-only PARSeq false-positive / confidence audit.

Parses sealed C3D JSON/JSONL artifacts only. No model load, inference,
threshold selection, or prediction mutation.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

ITEM_SCHEMA = "reid_jersey_parseq_false_positive_item_analysis_v1"
SUMMARY_SCHEMA = "reid_jersey_parseq_false_positive_audit_summary_v1"
MANIFEST_SCHEMA = "reid_jersey_parseq_false_positive_audit_manifest_v1"

POSITIVE_CLASS = "POS_readable"
GROUP_POSITIVE_EXACT = "positive_exact"
GROUP_POSITIVE_WRONG = "positive_wrong"
GROUP_POSITIVE_NONE = "positive_no_prediction"
GROUP_NEG_EMISSION = "negative_emission"
GROUP_NEG_NONE = "negative_no_emission"

FINAL_ARTIFACTS = (
    "parseq_false_positive_item_analysis.jsonl",
    "parseq_confidence_operating_points.jsonl",
    "parseq_confidence_group_summary.json",
    "parseq_output_distribution_summary.json",
    "parseq_selection_class_summary.json",
    "parseq_false_positive_audit_summary.json",
    "parseq_false_positive_audit_manifest.json",
)


class FalsePositiveAuditError(RuntimeError):
    """Contract failure for the C3E audit."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise FalsePositiveAuditError(f"{path}:{line_no} not an object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_unit(value: Any, label: str) -> float:
    if value is None:
        raise FalsePositiveAuditError(f"missing {label}")
    number = float(value)
    if not math.isfinite(number):
        raise FalsePositiveAuditError(f"non-finite {label}: {value}")
    if number < 0.0 or number > 1.0:
        raise FalsePositiveAuditError(f"{label} out of [0,1]: {number}")
    return number


def percentile_nearest_rank(sorted_values: Sequence[float], pct: float) -> float:
    """Deterministic nearest-rank percentile on a pre-sorted ascending list."""
    if not sorted_values:
        raise FalsePositiveAuditError("empty values for percentile")
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    rank = math.ceil((pct / 100.0) * len(sorted_values)) - 1
    rank = min(max(rank, 0), len(sorted_values) - 1)
    return float(sorted_values[rank])


def confidence_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "sorted_values": [],
        }
    ordered = sorted(float(v) for v in values)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "standard_deviation": std,
        "p10": percentile_nearest_rank(ordered, 10),
        "p25": percentile_nearest_rank(ordered, 25),
        "p75": percentile_nearest_rank(ordered, 75),
        "p90": percentile_nearest_rank(ordered, 90),
        "sorted_values": ordered,
    }


def assign_evaluation_group(selection_class: str, outcome: str) -> str:
    if selection_class == POSITIVE_CLASS:
        if outcome == "exact_match":
            return GROUP_POSITIVE_EXACT
        if outcome == "wrong_number":
            return GROUP_POSITIVE_WRONG
        if outcome == "no_prediction":
            return GROUP_POSITIVE_NONE
        raise FalsePositiveAuditError(f"unexpected positive outcome: {outcome}")
    if outcome == "number_emitted":
        return GROUP_NEG_EMISSION
    if outcome == "rejected":
        return GROUP_NEG_NONE
    raise FalsePositiveAuditError(f"unexpected negative outcome: {outcome}")


def join_predictions_and_evaluations(
    predictions: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    prediction_sha256: str,
    evaluation_sha256: str,
) -> list[dict[str, Any]]:
    if len(predictions) != 46 or len(evaluations) != 46:
        raise FalsePositiveAuditError("expected 46 prediction and evaluation rows")
    pred_ids = [row["review_item_id"] for row in predictions]
    eval_ids = [row["review_item_id"] for row in evaluations]
    if len(set(pred_ids)) != 46 or len(set(eval_ids)) != 46:
        raise FalsePositiveAuditError("duplicate review_item_id")
    if set(pred_ids) != set(eval_ids):
        raise FalsePositiveAuditError(
            f"join mismatch missing={sorted(set(eval_ids)-set(pred_ids))} "
            f"extra={sorted(set(pred_ids)-set(eval_ids))}"
        )
    eval_by_id = {row["review_item_id"]: row for row in evaluations}
    items: list[dict[str, Any]] = []
    for prediction in predictions:
        evaluation = eval_by_id[prediction["review_item_id"]]
        if int(prediction["pilot_index"]) != int(evaluation["pilot_index"]):
            raise FalsePositiveAuditError("pilot_index mismatch")
        if prediction["selection_class"] != evaluation["selection_class"]:
            raise FalsePositiveAuditError("selection_class mismatch")
        if prediction["source_crop_sha256"] != evaluation["source_crop_sha256"]:
            raise FalsePositiveAuditError("crop SHA mismatch on join")
        conf = _finite_unit(prediction.get("sequence_confidence"), "sequence_confidence")
        token_probs = prediction.get("token_probabilities") or []
        token_probs_f = [_finite_unit(v, "token_probability") for v in token_probs]
        min_tok = min(token_probs_f) if token_probs_f else None
        group = assign_evaluation_group(
            prediction["selection_class"], evaluation["outcome"]
        )
        accepted = evaluation.get("accepted_prediction")
        manual = (evaluation.get("manual_jersey_number") or "").strip()
        if group == GROUP_POSITIVE_EXACT:
            correctness = "exact"
        elif group == GROUP_POSITIVE_WRONG:
            correctness = "wrong"
        elif group == GROUP_POSITIVE_NONE:
            correctness = "no_prediction"
        elif group == GROUP_NEG_EMISSION:
            correctness = "negative_emission"
        else:
            correctness = "negative_no_emission"
        ref_len = None
        if manual:
            ref_len = len(manual)
        elif evaluation.get("manual_digit_count") not in (None, ""):
            try:
                ref_len = int(evaluation["manual_digit_count"])
            except (TypeError, ValueError):
                ref_len = None
        pred_len = len(str(accepted)) if accepted is not None else 0
        items.append(
            {
                "schema_version": ITEM_SCHEMA,
                "review_item_id": prediction["review_item_id"],
                "pilot_index": int(prediction["pilot_index"]),
                "selection_class": prediction["selection_class"],
                "evaluation_group": group,
                "reference_length": ref_len,
                "accepted_prediction": accepted,
                "prediction_length": pred_len,
                "prediction_correctness": correctness,
                "sequence_confidence": conf,
                "confidence_method": prediction.get("confidence_method"),
                "token_probabilities": token_probs_f,
                "minimum_selected_token_probability": min_tok,
                "eos_position": prediction.get("eos_position"),
                "rejection_reason": prediction.get("rejection_reason"),
                "source_prediction_sha256": prediction_sha256,
                "source_evaluation_sha256": evaluation_sha256,
                # analysis-only helpers (stripped before writing item jsonl if needed)
                "_manual_jersey_number": manual,
                "_outcome": evaluation["outcome"],
                "_first_token_probability": token_probs_f[0] if token_probs_f else None,
                "_second_token_probability": token_probs_f[1] if len(token_probs_f) > 1 else None,
            }
        )
    return items


def validate_group_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        GROUP_POSITIVE_EXACT: 0,
        GROUP_POSITIVE_WRONG: 0,
        GROUP_POSITIVE_NONE: 0,
        GROUP_NEG_EMISSION: 0,
        GROUP_NEG_NONE: 0,
    }
    for item in items:
        counts[item["evaluation_group"]] += 1
    if sum(counts.values()) != 46:
        raise FalsePositiveAuditError(f"group total != 46: {counts}")
    if counts[GROUP_POSITIVE_EXACT] != 5:
        raise FalsePositiveAuditError(f"expected 5 exact, got {counts[GROUP_POSITIVE_EXACT]}")
    if counts[GROUP_POSITIVE_WRONG] != 15:
        raise FalsePositiveAuditError(f"expected 15 wrong, got {counts[GROUP_POSITIVE_WRONG]}")
    if counts[GROUP_POSITIVE_NONE] != 0:
        raise FalsePositiveAuditError("expected 0 positive_no_prediction")
    if counts[GROUP_NEG_EMISSION] != 26:
        raise FalsePositiveAuditError(f"expected 26 negative emissions, got {counts[GROUP_NEG_EMISSION]}")
    if counts[GROUP_NEG_NONE] != 0:
        raise FalsePositiveAuditError("expected 0 negative_no_emission")
    return counts


def overlap_interval(a_min: float, a_max: float, b_min: float, b_max: float) -> Optional[list[float]]:
    lo = max(a_min, b_min)
    hi = min(a_max, b_max)
    if lo <= hi:
        return [lo, hi]
    return None


def pairwise_ordering_counts(
    left: Sequence[float], right: Sequence[float]
) -> dict[str, int]:
    greater = equal = less = 0
    for a in left:
        for b in right:
            if a > b:
                greater += 1
            elif a < b:
                less += 1
            else:
                equal += 1
    return {
        "left_greater": greater,
        "tie": equal,
        "left_less": less,
        "pair_count": greater + equal + less,
    }


def auroc_rank(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Deterministic Mann-Whitney AUROC with average ranks for ties.

    Returns the probability that a random positive score ranks above a random
    negative score (ties contribute 0.5). No external packages.
    """
    pos = [float(x) for x in positives]
    neg = [float(x) for x in negatives]
    if not pos or not neg:
        raise FalsePositiveAuditError("AUROC requires non-empty positive and negative sets")
    # All-equal shortcut
    if len(set(pos + neg)) == 1:
        return 0.5
    labeled = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    labeled.sort(key=lambda item: item[0])
    n = len(labeled)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labeled[j + 1][0] == labeled[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, labeled) if label == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def build_operating_points(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    confidences = sorted({float(item["sequence_confidence"]) for item in items})
    cuts: list[tuple[Optional[str], Optional[float]]] = [("accept_all", None)]
    for cut in confidences:
        cuts.append((None, cut))
    cuts.append(("accept_none", math.inf))

    rows: list[dict[str, Any]] = []
    for sentinel, cut in cuts:
        if sentinel == "accept_all":
            accepted = list(items)
            cut_value = None
        elif sentinel == "accept_none":
            accepted = []
            cut_value = None
        else:
            assert cut is not None
            accepted = [item for item in items if float(item["sequence_confidence"]) >= cut]
            cut_value = float(cut)

        exact_ret = sum(1 for item in accepted if item["evaluation_group"] == GROUP_POSITIVE_EXACT)
        wrong_ret = sum(1 for item in accepted if item["evaluation_group"] == GROUP_POSITIVE_WRONG)
        neg_ret = sum(1 for item in accepted if item["evaluation_group"] == GROUP_NEG_EMISSION)
        pos_total = sum(
            1
            for item in items
            if item["evaluation_group"]
            in {GROUP_POSITIVE_EXACT, GROUP_POSITIVE_WRONG, GROUP_POSITIVE_NONE}
        )
        neg_total = sum(
            1
            for item in items
            if item["evaluation_group"] in {GROUP_NEG_EMISSION, GROUP_NEG_NONE}
        )
        accepted_total = len(accepted)
        pos_accepted = exact_ret + wrong_ret
        rows.append(
            {
                "confidence_cut": cut_value,
                "sentinel_type": sentinel,
                "accepted_total": accepted_total,
                "exact_retained": exact_ret,
                "wrong_positive_retained": wrong_ret,
                "negative_retained": neg_ret,
                "positive_abstained": pos_total - pos_accepted,
                "negative_abstained": neg_total - neg_ret,
                "exact_fraction_of_accepted": (
                    exact_ret / accepted_total if accepted_total else None
                ),
                "negative_fraction_retained": neg_ret / neg_total if neg_total else None,
                "wrong_positive_fraction_retained": (
                    wrong_ret / 15 if 15 else None
                ),
                "selected": False,
            }
        )
    # Ensure no selected flag sneaks in
    if any(row.get("selected") is not False for row in rows):
        raise FalsePositiveAuditError("operating point selected flag must be false")
    if any("recommended_threshold" in row for row in rows):
        raise FalsePositiveAuditError("recommended_threshold forbidden")
    return rows


def frontier_metrics(operating_points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    zero_neg = [row for row in operating_points if int(row["negative_retained"]) == 0]
    all_exact = [row for row in operating_points if int(row["exact_retained"]) == 5]
    perfect = [
        row
        for row in operating_points
        if int(row["exact_retained"]) > 0
        and int(row["wrong_positive_retained"]) == 0
        and int(row["negative_retained"]) == 0
    ]
    exact_no_neg = [
        row
        for row in operating_points
        if int(row["negative_retained"]) == 0 and int(row["exact_retained"]) > 0
    ]

    def _cuts(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        return [row["confidence_cut"] for row in rows]

    zero_neg_max_exact = max((int(row["exact_retained"]) for row in zero_neg), default=0)
    zero_neg_best = [
        row for row in zero_neg if int(row["exact_retained"]) == zero_neg_max_exact
    ]
    zero_neg_min_wrong = (
        min(int(row["wrong_positive_retained"]) for row in zero_neg_best)
        if zero_neg_best
        else None
    )
    zero_neg_cuts = [
        row["confidence_cut"]
        for row in zero_neg_best
        if zero_neg_min_wrong is not None
        and int(row["wrong_positive_retained"]) == zero_neg_min_wrong
    ]

    all_exact_min_neg = (
        min(int(row["negative_retained"]) for row in all_exact) if all_exact else None
    )
    all_exact_at_min_neg = [
        row
        for row in all_exact
        if all_exact_min_neg is not None and int(row["negative_retained"]) == all_exact_min_neg
    ]
    all_exact_min_wrong = (
        min(int(row["wrong_positive_retained"]) for row in all_exact_at_min_neg)
        if all_exact_at_min_neg
        else None
    )
    all_exact_cuts = [
        row["confidence_cut"]
        for row in all_exact_at_min_neg
        if all_exact_min_wrong is not None
        and int(row["wrong_positive_retained"]) == all_exact_min_wrong
    ]

    return {
        "zero_negative_frontier": {
            "operating_point_count": len(zero_neg),
            "maximum_exact_retained": zero_neg_max_exact,
            "minimum_wrong_positive_retained_at_max_exact": zero_neg_min_wrong,
            "corresponding_cut_values": zero_neg_cuts,
            "note": "descriptive frontier only; not a selected threshold",
        },
        "all_exact_retention_frontier": {
            "operating_point_count": len(all_exact),
            "minimum_negative_retained": all_exact_min_neg,
            "minimum_wrong_positive_retained_at_min_negative": all_exact_min_wrong,
            "corresponding_cut_values": all_exact_cuts,
            "note": "descriptive frontier only; not a selected threshold",
        },
        "any_exact_zero_wrong_zero_negative": bool(perfect),
        "perfect_safe_point_cuts": _cuts(perfect),
        "exact_signal_without_negative": bool(exact_no_neg),
        "exact_signal_without_negative_cuts": _cuts(exact_no_neg),
    }


def frequency(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_output_distribution(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        by_group.setdefault(item["evaluation_group"], []).append(item)

    def _group_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        preds = [row.get("accepted_prediction") for row in rows]
        lengths = [int(row.get("prediction_length") or 0) for row in rows]
        eos = [row.get("eos_position") for row in rows]
        first = [row.get("_first_token_probability") for row in rows if row.get("_first_token_probability") is not None]
        second = [
            row.get("_second_token_probability")
            for row in rows
            if row.get("_second_token_probability") is not None
        ]
        mins = [
            row.get("minimum_selected_token_probability")
            for row in rows
            if row.get("minimum_selected_token_probability") is not None
        ]
        return {
            "count": len(rows),
            "predicted_number_frequency": frequency(preds),
            "one_digit_output_count": sum(1 for length in lengths if length == 1),
            "two_digit_output_count": sum(1 for length in lengths if length == 2),
            "prediction_length_distribution": frequency(lengths),
            "eos_position_distribution": frequency(eos),
            "first_token_probability": confidence_summary([float(v) for v in first]),
            "second_token_probability": confidence_summary([float(v) for v in second]),
            "minimum_selected_token_probability": confidence_summary([float(v) for v in mins]),
            "repeated_prediction_top": list(frequency(preds).items())[:10],
        }

    exact_nums = {
        str(row["accepted_prediction"])
        for row in by_group.get(GROUP_POSITIVE_EXACT, [])
        if row.get("accepted_prediction") is not None
    }
    neg_nums = {
        str(row["accepted_prediction"])
        for row in by_group.get(GROUP_NEG_EMISSION, [])
        if row.get("accepted_prediction") is not None
    }
    wrong_nums = {
        str(row["accepted_prediction"])
        for row in by_group.get(GROUP_POSITIVE_WRONG, [])
        if row.get("accepted_prediction") is not None
    }
    return {
        "by_group": {name: _group_block(rows) for name, rows in by_group.items()},
        "negative_most_frequent_numbers": frequency(
            [row.get("accepted_prediction") for row in by_group.get(GROUP_NEG_EMISSION, [])]
        ),
        "wrong_positive_most_frequent_numbers": frequency(
            [row.get("accepted_prediction") for row in by_group.get(GROUP_POSITIVE_WRONG, [])]
        ),
        "exact_number_list": sorted(exact_nums),
        "numbers_in_both_exact_and_negative": sorted(exact_nums & neg_nums),
        "numbers_in_both_wrong_and_negative": sorted(wrong_nums & neg_nums),
        "bias_flags_descriptive": {
            "output_length_bias_note": "inspect one_digit vs two_digit counts; no automatic correction",
            "single_digit_bias_candidate": (
                sum(1 for row in items if int(row.get("prediction_length") or 0) == 1)
                > sum(1 for row in items if int(row.get("prediction_length") or 0) == 2)
            ),
            "repeated_number_bias_candidate": True,
            "high_confidence_wrong_prediction_present": False,  # filled later
            "high_confidence_negative_emission_present": False,  # filled later
        },
    }


def build_selection_class_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    negatives = [
        item for item in items if item["selection_class"] != POSITIVE_CLASS
    ]
    if len(negatives) != 26:
        raise FalsePositiveAuditError("expected 26 negatives")
    classes = [
        "A_not_visible",
        "B_visible_unreadable",
        "C_uncertain_signal",
        "D_uncertain_crop",
        "E_invalid",
    ]
    by_class = {}
    for name in classes:
        rows = [item for item in negatives if item["selection_class"] == name]
        by_class[name] = {
            "item_count": len(rows),
            "accepted_emission_count": sum(
                1 for row in rows if row["evaluation_group"] == GROUP_NEG_EMISSION
            ),
            "emitted_numbers": frequency([row.get("accepted_prediction") for row in rows]),
            "confidence_summary": confidence_summary(
                [float(row["sequence_confidence"]) for row in rows]
            ),
            "output_length_distribution": frequency(
                [int(row.get("prediction_length") or 0) for row in rows]
            ),
            "eos_position_distribution": frequency([row.get("eos_position") for row in rows]),
        }
    positives = [item for item in items if item["selection_class"] == POSITIVE_CLASS]
    one_digit = [item for item in positives if item.get("reference_length") == 1]
    two_digit = [item for item in positives if item.get("reference_length") == 2]

    def _ref_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "item_count": len(rows),
            "exact_count": sum(1 for row in rows if row["evaluation_group"] == GROUP_POSITIVE_EXACT),
            "wrong_count": sum(1 for row in rows if row["evaluation_group"] == GROUP_POSITIVE_WRONG),
        }

    per_jersey: dict[str, dict[str, Any]] = {}
    for item in positives:
        jersey = str(item.get("_manual_jersey_number") or "")
        entry = per_jersey.setdefault(
            jersey,
            {"item_count": 0, "exact_count": 0, "wrong_count": 0, "predictions": []},
        )
        entry["item_count"] += 1
        entry["predictions"].append(item.get("accepted_prediction"))
        if item["evaluation_group"] == GROUP_POSITIVE_EXACT:
            entry["exact_count"] += 1
        elif item["evaluation_group"] == GROUP_POSITIVE_WRONG:
            entry["wrong_count"] += 1

    return {
        "negative_total": len(negatives),
        "by_selection_class": by_class,
        "positive_one_digit_reference": _ref_stats(one_digit),
        "positive_two_digit_reference": _ref_stats(two_digit),
        "per_manual_jersey_number_descriptive": per_jersey,
        "note": "frozen 46-item descriptive only; not a general benchmark",
    }


def build_evidence_and_decision(
    *,
    exact_count: int,
    negative_emission: int,
    frontiers: Mapping[str, Any],
    auroc_exact_vs_neg: float,
    overlap_exact_neg: Optional[list[float]],
    high_conf_neg: bool,
    high_conf_wrong: bool,
    confidence_analyzable: bool,
) -> tuple[list[str], str]:
    labels: list[str] = []
    if exact_count == 5:
        labels.append("PARSEQ_RECOGNITION_SIGNAL_PRESENT")
    if negative_emission == 26:
        labels.append("PARSEQ_RECOGNIZER_ONLY_ALWAYS_EMITS_ON_NEGATIVE_SET")
        labels.append("NO_NATURAL_ABSTENTION_OBSERVED")
    elif negative_emission < 26:
        labels.append("SOME_NATURAL_ABSTENTION_OBSERVED")

    zero_max_exact = int(
        frontiers["zero_negative_frontier"]["maximum_exact_retained"]
    )
    if zero_max_exact > 0:
        labels.append("CONFIDENCE_ZERO_NEGATIVE_FRONTIER_RETAINS_EXACT_SIGNAL")
    else:
        labels.append("CONFIDENCE_ZERO_NEGATIVE_FRONTIER_LOSES_ALL_EXACT_SIGNAL")

    if frontiers["any_exact_zero_wrong_zero_negative"]:
        labels.append("CONFIDENCE_PERFECT_SAFE_POINT_OBSERVED_IN_FROZEN_SET")
    else:
        labels.append("NO_CONFIDENCE_PERFECT_SAFE_POINT_IN_FROZEN_SET")

    if auroc_exact_vs_neg > 0.5:
        labels.append("CONFIDENCE_HAS_DESCRIPTIVE_RANKING_SIGNAL")
    elif auroc_exact_vs_neg == 0.5:
        labels.append("CONFIDENCE_NO_DESCRIPTIVE_RANKING_SIGNAL")
    else:
        labels.append("CONFIDENCE_RANKING_INVERTED_ON_FROZEN_SET")

    if overlap_exact_neg is not None:
        labels.append("EXACT_AND_NEGATIVE_CONFIDENCE_OVERLAP_PRESENT")
    if high_conf_neg:
        labels.append("HIGH_CONFIDENCE_NEGATIVE_EMISSION_PRESENT")
    if high_conf_wrong:
        labels.append("HIGH_CONFIDENCE_WRONG_POSITIVE_PRESENT")

    # Decision priority
    if not confidence_analyzable:
        decision = "GO_STAGE5C_C3F_LEGIBILITY_MODEL_CAPABILITY_AUDIT"
    elif frontiers["any_exact_zero_wrong_zero_negative"]:
        decision = "GO_STAGE5C_C3F_CONFIDENCE_GATE_VALIDATION"
    elif zero_max_exact > 0:
        decision = "GO_STAGE5C_C3F_CONFIDENCE_PLUS_LEGIBILITY_AUDIT"
    else:
        decision = "GO_STAGE5C_C3F_LEGIBILITY_MODEL_CAPABILITY_AUDIT"
    return labels, decision


def public_item_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ITEM_SCHEMA,
        "review_item_id": item["review_item_id"],
        "pilot_index": item["pilot_index"],
        "selection_class": item["selection_class"],
        "evaluation_group": item["evaluation_group"],
        "reference_length": item["reference_length"],
        "accepted_prediction": item["accepted_prediction"],
        "prediction_length": item["prediction_length"],
        "prediction_correctness": item["prediction_correctness"],
        "sequence_confidence": item["sequence_confidence"],
        "confidence_method": item["confidence_method"],
        "token_probabilities": item["token_probabilities"],
        "minimum_selected_token_probability": item["minimum_selected_token_probability"],
        "eos_position": item["eos_position"],
        "rejection_reason": item["rejection_reason"],
        "source_prediction_sha256": item["source_prediction_sha256"],
        "source_evaluation_sha256": item["source_evaluation_sha256"],
    }


def run_audit(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    c3d = _PROJECT_ROOT / config["c3d_output"]["path"]
    pred_path = c3d / "parseq_predictions.jsonl"
    eval_path = c3d / "parseq_item_evaluation.jsonl"
    run_manifest_path = c3d / "parseq_run_manifest.json"
    results_path = c3d / "parseq_results_summary.json"
    contract_path = c3d / "parseq_checkpoint_contract.json"

    # Verify C3D artifact SHAs from run manifest
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for name, meta in run_manifest["artifacts"].items():
        path = c3d / name
        actual = sha256_file(path)
        if actual != meta["sha256"]:
            raise FalsePositiveAuditError(f"C3D artifact drift: {name}")
        if path.stat().st_size != meta["byte_size"]:
            raise FalsePositiveAuditError(f"C3D size drift: {name}")

    pred_sha = sha256_file(pred_path)
    eval_sha = sha256_file(eval_path)
    if pred_sha != config["c3d_output"]["predictions_sha256"]:
        raise FalsePositiveAuditError("predictions sha mismatch vs config pin")
    if eval_sha != config["c3d_output"]["item_evaluation_sha256"]:
        raise FalsePositiveAuditError("evaluation sha mismatch vs config pin")

    predictions = load_jsonl(pred_path)
    evaluations = load_jsonl(eval_path)
    items = join_predictions_and_evaluations(
        predictions,
        evaluations,
        prediction_sha256=pred_sha,
        evaluation_sha256=eval_sha,
    )
    group_counts = validate_group_counts(items)

    methods = {item["confidence_method"] for item in items}
    if len(methods) != 1:
        raise FalsePositiveAuditError(f"inconsistent confidence methods: {methods}")
    confidence_method = next(iter(methods))

    exact = [item for item in items if item["evaluation_group"] == GROUP_POSITIVE_EXACT]
    wrong = [item for item in items if item["evaluation_group"] == GROUP_POSITIVE_WRONG]
    neg = [item for item in items if item["evaluation_group"] == GROUP_NEG_EMISSION]
    all_pos = exact + wrong
    all_non_exact = [item for item in items if item["evaluation_group"] != GROUP_POSITIVE_EXACT]

    exact_c = [float(item["sequence_confidence"]) for item in exact]
    wrong_c = [float(item["sequence_confidence"]) for item in wrong]
    neg_c = [float(item["sequence_confidence"]) for item in neg]
    pos_c = [float(item["sequence_confidence"]) for item in all_pos]
    non_exact_c = [float(item["sequence_confidence"]) for item in all_non_exact]

    group_summary = {
        "confidence_method": confidence_method,
        "group_counts": group_counts,
        "positive_exact": confidence_summary(exact_c),
        "positive_wrong": confidence_summary(wrong_c),
        "negative_emission": confidence_summary(neg_c),
        "all_positive": confidence_summary(pos_c),
        "all_non_exact": confidence_summary(non_exact_c),
    }

    overlap_exact_neg = overlap_interval(min(exact_c), max(exact_c), min(neg_c), max(neg_c))
    overlap_exact_wrong = overlap_interval(
        min(exact_c), max(exact_c), min(wrong_c), max(wrong_c)
    )
    max_neg = max(neg_c)
    min_exact = min(exact_c)
    max_exact = max(exact_c)
    max_wrong = max(wrong_c)
    exact_above_max_neg = sum(1 for value in exact_c if value > max_neg)
    neg_above_min_exact = sum(1 for value in neg_c if value > min_exact)
    high_conf_neg = any(value >= max_exact for value in neg_c)
    high_conf_wrong = any(value >= max_exact for value in wrong_c)

    auroc_exact_neg = auroc_rank(exact_c, neg_c)
    auroc_exact_nonexact = auroc_rank(exact_c, non_exact_c)
    auroc_pos_neg = auroc_rank(pos_c, neg_c)

    operating_points = build_operating_points(items)
    frontiers = frontier_metrics(operating_points)
    output_dist = build_output_distribution(items)
    output_dist["bias_flags_descriptive"]["high_confidence_wrong_prediction_present"] = (
        high_conf_wrong
    )
    output_dist["bias_flags_descriptive"]["high_confidence_negative_emission_present"] = (
        high_conf_neg
    )
    selection_summary = build_selection_class_summary(items)

    labels, decision = build_evidence_and_decision(
        exact_count=5,
        negative_emission=26,
        frontiers=frontiers,
        auroc_exact_vs_neg=auroc_exact_neg,
        overlap_exact_neg=overlap_exact_neg,
        high_conf_neg=high_conf_neg,
        high_conf_wrong=high_conf_wrong,
        confidence_analyzable=True,
    )

    # High-confidence wrong/negative examples (descriptive)
    high_neg_examples = sorted(
        (
            {
                "review_item_id": item["review_item_id"],
                "accepted_prediction": item["accepted_prediction"],
                "sequence_confidence": item["sequence_confidence"],
                "selection_class": item["selection_class"],
            }
            for item in neg
            if float(item["sequence_confidence"]) >= max_exact
        ),
        key=lambda row: -row["sequence_confidence"],
    )
    high_wrong_examples = sorted(
        (
            {
                "review_item_id": item["review_item_id"],
                "accepted_prediction": item["accepted_prediction"],
                "manual_jersey_number": item["_manual_jersey_number"],
                "sequence_confidence": item["sequence_confidence"],
            }
            for item in wrong
            if float(item["sequence_confidence"]) >= max_exact
        ),
        key=lambda row: -row["sequence_confidence"],
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "completed",
        "source_stage": "Stage_5C_C3D",
        "input_counts": {
            "total": 46,
            "positive": 20,
            "negative": 26,
            "exact": 5,
            "wrong_positive": 15,
            "positive_no_prediction": 0,
            "negative_emission": 26,
            "negative_no_emission": 0,
        },
        "confidence_method": confidence_method,
        "group_distributions": group_summary,
        "overlap_metrics": {
            "exact_confidence_min": min_exact,
            "exact_confidence_max": max_exact,
            "wrong_positive_confidence_min": min(wrong_c),
            "wrong_positive_confidence_max": max_wrong,
            "negative_confidence_min": min(neg_c),
            "negative_confidence_max": max_neg,
            "exact_vs_negative_overlap_interval": overlap_exact_neg,
            "exact_vs_wrong_overlap_interval": overlap_exact_wrong,
            "exact_count_above_max_negative": exact_above_max_neg,
            "negative_count_above_min_exact": neg_above_min_exact,
            "pairwise_exact_vs_negative": pairwise_ordering_counts(exact_c, neg_c),
            "pairwise_exact_vs_wrong": pairwise_ordering_counts(exact_c, wrong_c),
            "pairwise_wrong_vs_negative": pairwise_ordering_counts(wrong_c, neg_c),
        },
        "auroc_descriptive": {
            "exact_vs_negative": auroc_exact_neg,
            "exact_vs_all_non_exact": auroc_exact_nonexact,
            "readable_positive_vs_negative": auroc_pos_neg,
            "note": "small frozen set; no CI/significance claim",
        },
        "operating_point_frontiers": frontiers,
        "output_token_bias_metrics": output_dist["bias_flags_descriptive"],
        "high_confidence_examples_descriptive": {
            "negative_ge_max_exact": high_neg_examples,
            "wrong_positive_ge_max_exact": high_wrong_examples,
        },
        "natural_abstention": {
            "result": "NO_NATURAL_ABSTENTION_OBSERVED",
            "negative_emission": 26,
            "negative_no_emission": 0,
            "regex_rejection_count": sum(
                1 for item in items if item.get("rejection_reason") not in (None, "null")
            ),
            "empty_output_count": sum(
                1 for item in items if not (item.get("accepted_prediction") or "")
            ),
            "confidence_provides_descriptive_ranking": auroc_exact_neg > 0.5,
        },
        "evidence_labels": labels,
        "decision_class": decision,
        "interpretation_limits": [
            "descriptive frozen-set audit only",
            "no threshold was selected",
            "confidence is not treated as calibrated probability",
            "high confidence is not proof of correctness",
            "AUROC has no significance claim on n=46",
            "full pipeline not recommended from this gate alone",
        ],
        "recommended_full_pipeline": "not_recommended",
        "recommended_next_component": "SoccerNet_finetuned_legibility_classifier",
        "next_gate": decision,
        "safety_flags": {
            "model_initialized": False,
            "checkpoint_loaded": False,
            "inference_performed": False,
            "prediction_modified": False,
            "threshold_selected": False,
            "calibration_performed": False,
            "dataset_downloaded": False,
            "legibility_checkpoint_downloaded": False,
            "identity_assigned": False,
            "accuracy_claimed": False,
        },
        "source_c3d": {
            "path": str(c3d),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "predictions_sha256": pred_sha,
            "item_evaluation_sha256": eval_sha,
            "results_summary_sha256": sha256_file(results_path),
            "checkpoint_contract_sha256": sha256_file(contract_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / FINAL_ARTIFACTS[0], [public_item_row(item) for item in items])
    write_jsonl(output_dir / FINAL_ARTIFACTS[1], operating_points)
    write_json(output_dir / FINAL_ARTIFACTS[2], group_summary)
    write_json(output_dir / FINAL_ARTIFACTS[3], output_dist)
    write_json(output_dir / FINAL_ARTIFACTS[4], selection_summary)
    write_json(output_dir / FINAL_ARTIFACTS[5], summary)
    return summary


def finalize_manifest(
    config: Mapping[str, Any],
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    git_before: str,
    git_after: str,
) -> None:
    def row_count(path: Path) -> Optional[int]:
        if path.suffix == ".jsonl":
            return sum(1 for line in path.read_text().splitlines() if line.strip())
        return None

    artifacts = {}
    for name in FINAL_ARTIFACTS:
        if name == FINAL_ARTIFACTS[-1]:
            continue
        path = output_dir / name
        artifacts[name] = {
            "path": str(
                (_PROJECT_ROOT / config["output"]["final_dir"] / name).resolve()
            ),
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
            "row_count": row_count(path),
        }

    head = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(),
        "project_git_head": head,
        "source_c3d_path": str((_PROJECT_ROOT / config["c3d_output"]["path"]).resolve()),
        "source_c3d_run_manifest_sha256": summary["source_c3d"]["run_manifest_sha256"],
        "source_predictions_sha256": summary["source_c3d"]["predictions_sha256"],
        "source_evaluation_sha256": summary["source_c3d"]["item_evaluation_sha256"],
        "artifacts": artifacts,
        "git_status_before": git_before,
        "git_status_after_note": git_after,
        "source_immutability": {
            "c3d_artifacts_unmodified": True,
            "predictions_unmodified": True,
            "checkpoint_unmodified": True,
        },
        "no_model_no_network": True,
        "network_policy": "pass_offline_artifact_analysis_only",
        "atomic_finalization": True,
        "temp_cleanup": True,
        "decision_class": summary["decision_class"],
        "evidence_labels": summary["evidence_labels"],
        "safety_flags": summary["safety_flags"],
    }
    write_json(output_dir / FINAL_ARTIFACTS[-1], manifest)
    if len(list(output_dir.iterdir())) != 7:
        raise FalsePositiveAuditError("expected exactly 7 artifacts")


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "reid_jersey_parseq_false_positive_audit_config_v1":
        raise FalsePositiveAuditError("unexpected config schema")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    final_dir = _PROJECT_ROOT / config["output"]["final_dir"]
    if final_dir.exists():
        raise FalsePositiveAuditError(f"final output exists (no overwrite): {final_dir}")

    git_before = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    unique = f"{int(time.time())}_{os.getpid()}"
    temp_parent = _PROJECT_ROOT / "outputs" / "reid" / "full_stage4b"
    temp_dir = temp_parent / f"_tmp_jersey_parseq_fp_audit_stage5c_c3e_{unique}"
    work_home = Path(tempfile.mkdtemp(prefix="c3e_home_"))
    try:
        os.environ.update(
            {
                "HOME": str(work_home),
                "XDG_CACHE_HOME": str(work_home / "cache"),
                "TORCH_HOME": str(work_home / "torch"),
                "TMPDIR": str(work_home / "tmp"),
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
            }
        )
        (work_home / "cache").mkdir()
        (work_home / "torch").mkdir()
        (work_home / "tmp").mkdir()

        summary = run_audit(config, temp_dir)
        git_after = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        finalize_manifest(
            config,
            temp_dir,
            summary,
            git_before=git_before,
            git_after=git_after,
        )
        os.replace(temp_dir, final_dir)
        print(
            f"C3E_OK decision={summary['decision_class']} "
            f"zero_neg_exact={summary['operating_point_frontiers']['zero_negative_frontier']['maximum_exact_retained']} "
            f"auroc_exact_neg={summary['auroc_descriptive']['exact_vs_negative']:.4f}",
            flush=True,
        )
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work_home, ignore_errors=True)


if __name__ == "__main__":
    main()
