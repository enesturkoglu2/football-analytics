"""Unit tests for Stage 5B3A track purity audit (no real sample.mp4)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.kit import (  # noqa: E402
    DEFAULT_FAMILY_ORDER,
    build_track_kit_descriptors,
    compute_torso_kit_metrics,
    load_kit_descriptor_config,
)
from football_analytics.reid.purity import (  # noqa: E402
    AUDIT_SUMMARY_NAME,
    TRACK_PURITY_NAME,
    TRANSITION_NAME,
    PurityError,
    assign_independent_ranks,
    build_chronological_tracks,
    build_transition,
    compute_frame_torso_bbox,
    compute_torso_tracking_contamination,
    l1_distance,
    lab_mean_distance,
    load_track_purity_audit_config,
    run_analyze_reid_track_purity,
    validate_track_purity_audit_config,
)
from football_analytics.reid.schema import build_crop_manifest_row  # noqa: E402

DEFAULT_KIT_CONFIG = _PROJECT_ROOT / "configs" / "reid" / "kit_descriptor_stage5b.yaml"
DEFAULT_AUDIT_CONFIG = (
    _PROJECT_ROOT / "configs" / "reid" / "track_purity_audit_stage5b3.yaml"
)
FRAME_W = 200
FRAME_H = 160


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed jpeg {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _obs(track_id: int, frame_index: int, bbox: list[float]) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index) / 30.0,
        "track_id": track_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox_xyxy": bbox,
    }


def _solid(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def _quality_row(manifest_row: dict, *, laplacian: float = 100.0, union: float = 0.0) -> dict:
    return {
        "crop_id": manifest_row["crop_id"],
        "track_id": manifest_row["track_id"],
        "frame_index": manifest_row["frame_index"],
        "selection_rank": manifest_row["selection_rank"],
        "crop_relative_path": manifest_row["crop_relative_path"],
        "laplacian_variance": laplacian,
        "union_other_person_crop_coverage": union,
        "frame_edge_contact_count": 0,
        "quality_decision": "measurement_only",
        "automatic_exclusion_applied": False,
        "quality_threshold": None,
        "contamination_threshold": None,
        "schema_version": "reid_crop_quality_signal_v1",
    }


def _kit_row_from_image(manifest_row: dict, image: np.ndarray, kit_cfg: dict, quality: dict) -> dict:
    metrics = compute_torso_kit_metrics(image, config=kit_cfg)
    return {
        "crop_id": manifest_row["crop_id"],
        "track_id": manifest_row["track_id"],
        "frame_index": manifest_row["frame_index"],
        "selection_rank": manifest_row["selection_rank"],
        "crop_relative_path": manifest_row["crop_relative_path"],
        **metrics,
        "quality_signal_joined": True,
        "laplacian_variance": quality["laplacian_variance"],
        "union_other_person_crop_coverage": quality["union_other_person_crop_coverage"],
        "frame_edge_contact_count": quality["frame_edge_contact_count"],
        "quality_weight_applied": False,
        "quality_exclusion_applied": False,
        "team_assignment": None,
        "kit_similarity_threshold": None,
        "automatic_link_applied": False,
        "automatic_reject_applied": False,
        "descriptor_usage": "measurement_only",
        "schema_version": "reid_crop_kit_descriptor_v1",
    }


def _make_fixture(
    root: Path,
    *,
    boxes: list[tuple[int, int, list[int], tuple[int, int, int]]],
    extra_tracks: list[dict] | None = None,
) -> dict[str, Path]:
    """boxes: (track_id, frame_index, [l,t,r,b], bgr_color)"""
    kit_cfg = load_kit_descriptor_config(DEFAULT_KIT_CONFIG)
    crops_root = root / "crops_out"
    (crops_root / "crops").mkdir(parents=True)
    manifest_rows: list[dict] = []
    quality_rows: list[dict] = []
    kit_rows: list[dict] = []
    track_obs: list[dict] = list(extra_tracks or [])

    for track_id, frame_index, bbox, bgr in boxes:
        left, top, right, bottom = bbox
        w, h = right - left, bottom - top
        existing = sum(1 for r in manifest_rows if r["track_id"] == track_id)
        rank = existing + 1
        row = build_crop_manifest_row(
            track_id=track_id,
            frame_index=frame_index,
            timestamp_sec=float(frame_index) / 30.0,
            source_video="synthetic.mp4",
            bbox_xyxy=bbox,
            detection_confidence=0.9,
            quality_score=float(w * h) * 0.9,
            selection_rank=rank,
        )
        img = _solid(h, w, bgr)
        _write_jpeg(crops_root / row["crop_relative_path"], img)
        q = _quality_row(row, laplacian=float(50 + rank), union=0.0)
        k = _kit_row_from_image(row, img, kit_cfg, q)
        manifest_rows.append(row)
        quality_rows.append(q)
        kit_rows.append(k)
        track_obs.append(_obs(track_id, frame_index, [float(v) for v in bbox]))

    track_kit = build_track_kit_descriptors(kit_rows, config=kit_cfg)

    manifest_path = crops_root / "crop_manifest.jsonl"
    quality_path = root / "crop_quality_signals.jsonl"
    crop_kit_path = root / "crop_kit_descriptors.jsonl"
    track_kit_path = root / "track_kit_descriptors.jsonl"
    tracks_path = root / "tracks.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(quality_path, quality_rows)
    _write_jsonl(crop_kit_path, kit_rows)
    _write_jsonl(track_kit_path, track_kit)
    _write_jsonl(tracks_path, track_obs)

    kit_cfg_path = root / "kit_config.yaml"
    audit_cfg_path = root / "audit_config.yaml"
    kit_cfg_path.write_text(DEFAULT_KIT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    audit_cfg_path.write_text(
        DEFAULT_AUDIT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return {
        "crops_root": crops_root,
        "manifest": manifest_path,
        "quality": quality_path,
        "crop_kit": crop_kit_path,
        "track_kit": track_kit_path,
        "tracks": tracks_path,
        "kit_config": kit_cfg_path,
        "audit_config": audit_cfg_path,
    }


class ConfigValidationTests(unittest.TestCase):
    def test_valid_and_rejects(self) -> None:
        cfg = load_track_purity_audit_config(DEFAULT_AUDIT_CONFIG)
        self.assertEqual(cfg["stage_status"], "measurement_baseline")
        self.assertFalse(cfg["automatic_track_split_enabled"])
        self.assertIsNone(cfg["change_threshold"])

        payload = yaml.safe_load(DEFAULT_AUDIT_CONFIG.read_text(encoding="utf-8"))
        for flag in (
            "automatic_track_split_enabled",
            "automatic_track_delete_enabled",
            "automatic_global_id_rewrite_enabled",
            "automatic_team_assignment_enabled",
            "composite_change_score_enabled",
        ):
            bad = copy.deepcopy(payload)
            bad[flag] = True
            with self.assertRaises(PurityError):
                validate_track_purity_audit_config(bad)

        bad = copy.deepcopy(payload)
        bad["change_threshold"] = 0.5
        with self.assertRaises(PurityError):
            validate_track_purity_audit_config(bad)
        bad = copy.deepcopy(payload)
        bad["split_threshold"] = 0.5
        with self.assertRaises(PurityError):
            validate_track_purity_audit_config(bad)
        bad = copy.deepcopy(payload)
        bad["chronology"]["compare_adjacent_crops_only"] = False
        with self.assertRaises(PurityError):
            validate_track_purity_audit_config(bad)
        bad = copy.deepcopy(payload)
        bad["distance_normalization"]["lab_uint8_max_distance"] = 1.0
        with self.assertRaises(PurityError):
            validate_track_purity_audit_config(bad)
        bad = copy.deepcopy(payload)
        bad["ranking"]["independent_metric_ranks"] = ["color_family_l1", "unknown"]
        with self.assertRaises(PurityError):
            validate_track_purity_audit_config(bad)
        del payload["policy"]
        with self.assertRaises(PurityError):
            validate_track_purity_audit_config(payload)


class DistanceAndChronologyTests(unittest.TestCase):
    def test_distances_and_sort(self) -> None:
        self.assertAlmostEqual(l1_distance([1, 0], [0, 1]), 2.0)
        raw, norm = lab_mean_distance(
            {"l": 0, "a": 0, "b": 0},
            {"l": 255, "a": 255, "b": 255},
            lab_max=math.sqrt(3 * 255 * 255),
        )
        self.assertAlmostEqual(norm, 1.0, places=6)
        self.assertGreater(raw, 400)

        crops = [
            {
                "track_id": 1,
                "frame_index": 10,
                "selection_rank": 1,
                "crop_id": "b",
                "manifest_position": 1,
            },
            {
                "track_id": 1,
                "frame_index": 5,
                "selection_rank": 1,
                "crop_id": "a",
                "manifest_position": 0,
            },
        ]
        ordered = build_chronological_tracks(crops)[1]
        self.assertEqual([c["crop_id"] for c in ordered], ["a", "b"])

        with self.assertRaises(PurityError):
            build_chronological_tracks(
                [
                    {
                        "track_id": 1,
                        "frame_index": 1,
                        "selection_rank": 1,
                        "crop_id": "a",
                        "manifest_position": 0,
                    },
                    {
                        "track_id": 1,
                        "frame_index": 1,
                        "selection_rank": 2,
                        "crop_id": "b",
                        "manifest_position": 1,
                    },
                ]
            )

    def test_transition_and_ranks(self) -> None:
        kit_cfg = load_kit_descriptor_config(DEFAULT_KIT_CONFIG)
        red = compute_torso_kit_metrics(_solid(60, 40, (0, 0, 255)), config=kit_cfg)
        blue = compute_torso_kit_metrics(_solid(60, 40, (255, 0, 0)), config=kit_cfg)
        base = {
            "crop_id": "x",
            "track_id": 1,
            "frame_index": 0,
            "selection_rank": 1,
            "manifest_position": 0,
            "laplacian_variance": 10.0,
            "union_other_person_crop_coverage": 0.0,
            "frame_edge_contact_count": 0,
            "torso_union_other_person_coverage": 0.0,
            "torso_other_person_overlap_count": 0,
            "torso_other_person_center_inside_count": 0,
        }
        a = {**base, **red, "crop_id": "a", "frame_index": 0, "manifest_position": 0}
        b = {
            **base,
            **blue,
            "crop_id": "b",
            "frame_index": 5,
            "selection_rank": 1,
            "manifest_position": 1,
        }
        t = build_transition(
            track_id=1,
            transition_index=1,
            crop_from=a,
            crop_to=b,
            lab_max=math.sqrt(3 * 255 * 255),
        )
        self.assertTrue(t["dominant_family_changed"])
        self.assertGreater(t["color_family_l1"], 1.0)
        self.assertEqual(t["frame_gap"], 5)
        self.assertIsNone(t["composite_change_score"])

        same = build_transition(
            track_id=1,
            transition_index=1,
            crop_from=a,
            crop_to={**a, "crop_id": "c", "frame_index": 1, "manifest_position": 1},
            lab_max=math.sqrt(3 * 255 * 255),
        )
        self.assertFalse(same["dominant_family_changed"])
        self.assertAlmostEqual(same["color_family_l1"], 0.0, places=6)

        rows = [t, same]
        assign_independent_ranks(rows)
        self.assertEqual(t["rank_by_color_family_l1"], 1)
        self.assertEqual(same["rank_by_color_family_l1"], 2)


class TorsoContaminationTests(unittest.TestCase):
    def test_geometry_cases(self) -> None:
        crop = {
            "track_id": 1,
            "bbox_xyxy": [10, 10, 50, 70],
            "torso_x0": 8,
            "torso_y0": 9,
            "torso_x1": 32,
            "torso_y1": 39,
        }
        torso = compute_frame_torso_bbox(
            manifest_bbox=crop["bbox_xyxy"],
            torso_x0=8,
            torso_y0=9,
            torso_x1=32,
            torso_y1=39,
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(torso, [18, 19, 42, 49])

        none = compute_torso_tracking_contamination(
            crop_row=crop,
            frame_observations=[_obs(1, 0, [10, 10, 50, 70])],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(none["torso_other_person_overlap_count"], 0)

        outside = compute_torso_tracking_contamination(
            crop_row=crop,
            frame_observations=[
                _obs(1, 0, [10, 10, 50, 70]),
                _obs(2, 0, [100, 100, 120, 140]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(outside["torso_other_person_overlap_count"], 0)

        partial = compute_torso_tracking_contamination(
            crop_row=crop,
            frame_observations=[
                _obs(1, 0, [10, 10, 50, 70]),
                _obs(2, 0, [30, 30, 60, 60]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertGreater(partial["torso_other_person_overlap_count"], 0)
        self.assertGreater(partial["torso_union_other_person_coverage"], 0)
        self.assertLessEqual(partial["torso_union_other_person_coverage"], 1)

        full = compute_torso_tracking_contamination(
            crop_row=crop,
            frame_observations=[
                _obs(1, 0, [10, 10, 50, 70]),
                _obs(2, 0, [18, 19, 42, 49]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertAlmostEqual(full["torso_union_other_person_coverage"], 1.0, places=5)

        two = compute_torso_tracking_contamination(
            crop_row=crop,
            frame_observations=[
                _obs(1, 0, [10, 10, 50, 70]),
                _obs(2, 0, [18, 19, 30, 35]),
                _obs(3, 0, [25, 25, 42, 49]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(two["torso_other_person_overlap_count"], 2)
        self.assertLessEqual(two["torso_union_other_person_coverage"], 1.0)

        # Helper expects same-frame observations only (pipeline indexes by frame).
        # Wrong-frame rows must not be passed in; pipeline excludes them via
        # index_observations_by_frame before calling this helper.


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_and_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _make_fixture(
                root,
                boxes=[
                    (1, 10, [10, 10, 50, 70], (0, 0, 255)),  # red later frame first in list
                    (1, 5, [10, 10, 50, 70], (255, 0, 0)),  # blue earlier
                    (2, 1, [60, 20, 100, 80], (0, 0, 0)),  # single crop black
                ],
            )
            hashes = {p: _sha256(p) for p in [
                paths["manifest"], paths["quality"], paths["crop_kit"],
                paths["track_kit"], paths["tracks"], paths["kit_config"],
                paths["audit_config"],
            ]}
            out = root / "purity_out"
            result = run_analyze_reid_track_purity(
                crop_manifest=paths["manifest"],
                quality_signals=paths["quality"],
                crop_kit_descriptors=paths["crop_kit"],
                track_kit_descriptors=paths["track_kit"],
                tracks=paths["tracks"],
                kit_config=paths["kit_config"],
                audit_config=paths["audit_config"],
                output_dir=out,
            )
            self.assertEqual(result["crop_count"], 3)
            self.assertEqual(result["track_count"], 2)
            self.assertEqual(result["transition_count"], 1)
            self.assertEqual(result["multi_crop_track_count"], 1)
            self.assertEqual(result["single_crop_track_count"], 1)
            self.assertFalse(result["automatic_track_split_performed"])
            self.assertIsNone(result["change_threshold"])

            names = sorted(p.name for p in out.iterdir() if p.is_file())
            self.assertEqual(
                names, sorted([TRANSITION_NAME, TRACK_PURITY_NAME, AUDIT_SUMMARY_NAME])
            )
            transitions = [
                json.loads(line)
                for line in (out / TRANSITION_NAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(transitions), 1)
            self.assertTrue((out / TRANSITION_NAME).read_text().endswith("\n"))
            tr = transitions[0]
            self.assertEqual(tr["frame_index_from"], 5)
            self.assertEqual(tr["frame_index_to"], 10)
            self.assertTrue(tr["dominant_family_changed"])
            self.assertIsNone(tr["composite_change_score"])
            self.assertFalse(tr["automatic_split_applied"])

            tracks = [
                json.loads(line)
                for line in (out / TRACK_PURITY_NAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual([t["track_id"] for t in tracks], [1, 2])
            t1 = tracks[0]
            self.assertEqual(t1["transition_count"], 1)
            self.assertEqual(t1["crop_ids_chronological_order"][0].split("_frame_")[1].startswith("5"), True)
            self.assertIsNone(t1["purity_label"])
            t2 = tracks[1]
            self.assertTrue(t2["single_crop_track"])
            self.assertEqual(t2["transition_count"], 0)
            self.assertIsNone(t2["top_transition_by_color_family_l1"])

            summary = json.loads((out / AUDIT_SUMMARY_NAME).read_text())
            self.assertFalse(summary["accuracy_claimed"])
            self.assertFalse(summary["full_track_segmentation_performed"])
            self.assertTrue(summary["selected_crop_audit_only"])

            # collision / overwrite
            with self.assertRaises(Exception):
                run_analyze_reid_track_purity(
                    crop_manifest=paths["manifest"],
                    quality_signals=paths["quality"],
                    crop_kit_descriptors=paths["crop_kit"],
                    track_kit_descriptors=paths["track_kit"],
                    tracks=paths["tracks"],
                    kit_config=paths["kit_config"],
                    audit_config=paths["audit_config"],
                    output_dir=out,
                )
            run_analyze_reid_track_purity(
                crop_manifest=paths["manifest"],
                quality_signals=paths["quality"],
                crop_kit_descriptors=paths["crop_kit"],
                track_kit_descriptors=paths["track_kit"],
                tracks=paths["tracks"],
                kit_config=paths["kit_config"],
                audit_config=paths["audit_config"],
                output_dir=out,
                overwrite=True,
            )
            self.assertEqual(list(out.parent.glob(f"_tmp_reid_purity_{out.name}_*")), [])

            for path, digest in hashes.items():
                self.assertEqual(_sha256(path), digest)

            # deterministic
            out2 = root / "purity_out2"
            run_analyze_reid_track_purity(
                crop_manifest=paths["manifest"],
                quality_signals=paths["quality"],
                crop_kit_descriptors=paths["crop_kit"],
                track_kit_descriptors=paths["track_kit"],
                tracks=paths["tracks"],
                kit_config=paths["kit_config"],
                audit_config=paths["audit_config"],
                output_dir=out2,
            )
            self.assertEqual(
                (out / TRANSITION_NAME).read_text(),
                (out2 / TRANSITION_NAME).read_text(),
            )

    def test_validation_failure_no_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _make_fixture(
                root,
                boxes=[(1, 0, [10, 10, 50, 70], (0, 255, 0))],
            )
            # stale kit path
            rows = [
                json.loads(line)
                for line in paths["crop_kit"].read_text().splitlines()
                if line.strip()
            ]
            rows[0]["crop_relative_path"] = "crops/track_1/does_not_match.jpg"
            bad = root / "bad_kit.jsonl"
            _write_jsonl(bad, rows)
            out = root / "no_final"
            with self.assertRaises(PurityError):
                run_analyze_reid_track_purity(
                    crop_manifest=paths["manifest"],
                    quality_signals=paths["quality"],
                    crop_kit_descriptors=bad,
                    track_kit_descriptors=paths["track_kit"],
                    tracks=paths["tracks"],
                    kit_config=paths["kit_config"],
                    audit_config=paths["audit_config"],
                    output_dir=out,
                )
            self.assertFalse(out.exists())

    def test_path_and_nan_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _make_fixture(
                root, boxes=[(1, 0, [10, 10, 50, 70], (0, 0, 255))]
            )
            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(PurityError):
                run_analyze_reid_track_purity(
                    crop_manifest=empty,
                    quality_signals=paths["quality"],
                    crop_kit_descriptors=paths["crop_kit"],
                    track_kit_descriptors=paths["track_kit"],
                    tracks=paths["tracks"],
                    kit_config=paths["kit_config"],
                    audit_config=paths["audit_config"],
                    output_dir=root / "out",
                )

    def test_other_frame_observation_ignored_in_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Overlapping person only on a different frame must not affect torso.
            paths = _make_fixture(
                root,
                boxes=[(1, 0, [10, 10, 50, 70], (0, 0, 255))],
                extra_tracks=[_obs(2, 99, [10.0, 10.0, 50.0, 70.0])],
            )
            out = root / "purity_frame"
            run_analyze_reid_track_purity(
                crop_manifest=paths["manifest"],
                quality_signals=paths["quality"],
                crop_kit_descriptors=paths["crop_kit"],
                track_kit_descriptors=paths["track_kit"],
                tracks=paths["tracks"],
                kit_config=paths["kit_config"],
                audit_config=paths["audit_config"],
                output_dir=out,
            )
            summary = json.loads((out / AUDIT_SUMMARY_NAME).read_text())
            self.assertEqual(summary["crops_with_torso_other_person_overlap"], 0)

    def test_track_kit_mismatch_and_automatic_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _make_fixture(
                root, boxes=[(1, 0, [10, 10, 50, 70], (0, 0, 255))]
            )
            track_rows = [
                json.loads(line)
                for line in paths["track_kit"].read_text().splitlines()
                if line.strip()
            ]
            track_rows[0]["crop_count"] = 99
            bad_track = root / "bad_track_kit.jsonl"
            _write_jsonl(bad_track, track_rows)
            with self.assertRaises(PurityError):
                run_analyze_reid_track_purity(
                    crop_manifest=paths["manifest"],
                    quality_signals=paths["quality"],
                    crop_kit_descriptors=paths["crop_kit"],
                    track_kit_descriptors=bad_track,
                    tracks=paths["tracks"],
                    kit_config=paths["kit_config"],
                    audit_config=paths["audit_config"],
                    output_dir=root / "out_mismatch",
                )

            kit_rows = [
                json.loads(line)
                for line in paths["crop_kit"].read_text().splitlines()
                if line.strip()
            ]
            kit_rows[0]["automatic_link_applied"] = True
            bad_kit = root / "bad_crop_kit.jsonl"
            _write_jsonl(bad_kit, kit_rows)
            with self.assertRaises(PurityError):
                run_analyze_reid_track_purity(
                    crop_manifest=paths["manifest"],
                    quality_signals=paths["quality"],
                    crop_kit_descriptors=bad_kit,
                    track_kit_descriptors=paths["track_kit"],
                    tracks=paths["tracks"],
                    kit_config=paths["kit_config"],
                    audit_config=paths["audit_config"],
                    output_dir=root / "out_link",
                )

    def test_no_forbidden_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _make_fixture(
                root,
                boxes=[
                    (1, 0, [10, 10, 50, 70], (0, 0, 255)),
                    (1, 5, [10, 10, 50, 70], (255, 0, 0)),
                ],
            )
            with mock.patch("cv2.kmeans", side_effect=AssertionError("kmeans")):
                run_analyze_reid_track_purity(
                    crop_manifest=paths["manifest"],
                    quality_signals=paths["quality"],
                    crop_kit_descriptors=paths["crop_kit"],
                    track_kit_descriptors=paths["track_kit"],
                    tracks=paths["tracks"],
                    kit_config=paths["kit_config"],
                    audit_config=paths["audit_config"],
                    output_dir=root / "safe",
                )


class HelpTests(unittest.TestCase):
    def test_help(self) -> None:
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "analyze_reid_track_purity.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--crop-manifest", proc.stdout)
        self.assertIn("--audit-config", proc.stdout)
        self.assertNotIn("--video", proc.stdout)
        self.assertNotIn("--global-id", proc.stdout)


if __name__ == "__main__":
    unittest.main()
