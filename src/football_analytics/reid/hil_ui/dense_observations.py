"""Dense per-frame bbox observations for interactive video overlay (no tracking rerun)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_dense_observations_from_mapping(
    mapping_rows: Sequence[Mapping[str, Any]],
    *,
    codes: Sequence[str],
    segment_id_fn,
) -> dict[str, list[dict[str, Any]]]:
    """Return frame_index(str) → list of bbox observation dicts."""
    by_code = {str(r["external_candidate_code"]): r for r in mapping_rows}
    out: dict[str, list[dict[str, Any]]] = {}
    for code in codes:
        row = by_code.get(code)
        if row is None:
            continue
        seg = segment_id_fn(code)
        raw = str(row["raw_external_track_id"])
        for item in row.get("bbox_per_observation") or []:
            fi = int(item["frame_index"])
            key = str(fi)
            out.setdefault(key, []).append(
                {
                    "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                    "segment_id": seg,
                    "raw_track_id": raw,
                    "external_candidate_code": code,
                    "candidate_id": None,
                }
            )
    return out


def attach_candidate_ids(
    observations: dict[str, list[dict[str, Any]]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_seg = {str(c["segment_id"]): str(c["candidate_id"]) for c in candidates}
    for _fi, rows in observations.items():
        for row in rows:
            row["candidate_id"] = by_seg.get(str(row.get("segment_id")))
    return observations


def load_mapping_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_dense_observation_json(path: Path, observations: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
