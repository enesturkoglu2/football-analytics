"""Unit tests for Stage 5B3F-A non-destructive manual segment view."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.segments import (  # noqa: E402
    SEGMENT_OBS_NAME,
    SEGMENTS_NAME,
    SUMMARY_NAME,
    UNASSIGNED_NAME,
    SegmentError,
    build_segment_view,
    canonical_observation_json,
    load_raw_track_observations,
    load_segment_decisions,
    load_segmentation_policy,
    run_build_manual_track_segment_view,
    source_observation_sha256,
    validate_segment_decisions,
    validate_segmentation_policy,
)

DEFAULT_POLICY = _PROJECT_ROOT / "configs" / "reid" / "manual_track_segmentation_policy_stage5b3.yaml"


def _obs(
    track_id: int,
    frame_index: int,
    *,
    bbox: list[float] | None = None,
    confidence: float = 0.9,
) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index) / 30.0,
        "track_id": track_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": confidence,
        "bbox_xyxy": bbox or [10.0, 20.0, 40.0, 80.0],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _base_policy() -> dict:
    return yaml.safe_load(DEFAULT_POLICY.read_text(encoding="utf-8"))


def _split_decision(
    *,
    raw_track_id: int = 10,
    segments: list[dict] | None = None,
    ambiguous: list[int] | None = None,
    gaps: list[list[int]] | None = None,
    boundaries: list[dict] | None = None,
) -> dict:
    if segments is None:
        segments = [
            {
                "segment_id": f"raw_{raw_track_id}_s01",
                "raw_track_id": raw_track_id,
                "frame_min": 0,
                "frame_max": 2,
                "include_existing_observations_only": True,
                "proven_physical_identity": False,
                "team_assignment": None,
                "global_id": None,
            },
            {
                "segment_id": f"raw_{raw_track_id}_s02",
                "raw_track_id": raw_track_id,
                "frame_min": 5,
                "frame_max": 7,
                "include_existing_observations_only": True,
                "proven_physical_identity": False,
                "team_assignment": None,
                "global_id": None,
            },
        ]
    if boundaries is None:
        boundaries = [
            {
                "last_segment_frame": 2,
                "next_segment_frame": 5,
                "boundary_type": "gap_bounded",
                "exact_real_world_switch_frame_known": False,
                "automatic_split_decision": False,
            }
        ]
    return {
        "raw_track_id": raw_track_id,
        "decision": "manual_split_candidate",
        "evidence_strength": "high",
        "probable_switch_event_count": len(boundaries),
        "manual_split_candidate": True,
        "ambiguous_existing_observation_frames": ambiguous or [],
        "unobserved_gap_ranges": gaps if gaps is not None else [[3, 4]],
        "segments": segments,
        "boundaries": boundaries,
        "notes": "synthetic",
    }


def _control_decision(raw_track_id: int = 20) -> dict:
    return {
        "raw_track_id": raw_track_id,
        "decision": "no_split_contamination_control",
        "evidence_strength": "medium",
        "probable_switch_event_count": 0,
        "manual_split_candidate": False,
        "raw_track_preserved": True,
        "ambiguous_existing_observation_frames": [],
        "unobserved_gap_ranges": [],
        "segments": [],
        "boundaries": [],
        "notes": "synthetic control",
    }


def _decisions_doc(tracks: list[dict]) -> dict:
    return {
        "schema_version": "reid_manual_track_segment_decisions_v1",
        "status": "visually_reviewed_plan_not_applied",
        "automatic_application_enabled": False,
        "manual_visual_review_is_ground_truth": False,
        "raw_tracks_mutated": False,
        "global_id_map_changed": False,
        "tracks": tracks,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PolicyValidationTests(unittest.TestCase):
    def test_valid_and_rejects(self) -> None:
        payload = _base_policy()
        validate_segmentation_policy(payload)

        cases = [
            ("raw_tracks", "immutable", False),
            ("raw_tracks", "source_observations_preserved", False),
            ("raw_tracks", "in_place_mutation_allowed", True),
            ("raw_tracks", "interpolation_allowed", True),
            ("raw_tracks", "deletion_allowed", True),
            ("segmentation", "automatic_split_enabled", True),
            ("segmentation", "automatic_boundary_selection_enabled", True),
            ("evaluation", "accuracy_claim_allowed", True),
            ("reid", "automatic_segment_merge_enabled", True),
            ("reid", "automatic_segment_link_enabled", True),
            ("global_identity", "global_id_rewrite_enabled", True),
        ]
        for section, key, bad in cases:
            bad_payload = copy.deepcopy(payload)
            bad_payload[section][key] = bad
            with self.assertRaises(SegmentError):
                validate_segmentation_policy(bad_payload)

        bad_payload = copy.deepcopy(payload)
        bad_payload["segmentation"]["frame_range_semantics"] = "all_frames"
        with self.assertRaises(SegmentError):
            validate_segmentation_policy(bad_payload)

        bad_payload = copy.deepcopy(payload)
        bad_payload["evaluation"]["manual_visual_observation_is_ground_truth"] = True
        with self.assertRaises(SegmentError):
            validate_segmentation_policy(bad_payload)


class DecisionValidationTests(unittest.TestCase):
    def test_valid_and_rejects(self) -> None:
        doc = _decisions_doc([_split_decision(), _control_decision()])
        validate_segment_decisions(doc)

        bad = copy.deepcopy(doc)
        bad["tracks"].append(_split_decision(raw_track_id=10))
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["segments"][1]["segment_id"] = bad["tracks"][0]["segments"][0][
            "segment_id"
        ]
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["segments"][1]["frame_min"] = 1
        bad["tracks"][0]["segments"][1]["frame_max"] = 3
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["segments"][0]["frame_max"] = -1
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["ambiguous_existing_observation_frames"] = [1]
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["boundaries"][0]["boundary_type"] = "exact_known"
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["boundaries"][0]["exact_real_world_switch_frame_known"] = True
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["boundaries"][0]["automatic_split_decision"] = True
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][1]["segments"] = [
            {
                "segment_id": "raw_20_s01",
                "raw_track_id": 20,
                "frame_min": 0,
                "frame_max": 1,
                "include_existing_observations_only": True,
                "proven_physical_identity": False,
                "team_assignment": None,
                "global_id": None,
            }
        ]
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["segments"] = bad["tracks"][0]["segments"][:1]
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["segments"][0]["proven_physical_identity"] = True
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)

        bad = copy.deepcopy(doc)
        bad["tracks"][0]["segments"][0]["team_assignment"] = "team_A"
        with self.assertRaises(SegmentError):
            validate_segment_decisions(bad)


class RawTrackingSchemaTests(unittest.TestCase):
    def test_valid_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "tracks.jsonl"
            rows = [_obs(1, 0), _obs(1, 1), _obs(2, 0)]
            _write_jsonl(path, rows)
            loaded = load_raw_track_observations(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["source_row_index"], 0)
            self.assertEqual(
                loaded[0]["source_observation_sha256"],
                source_observation_sha256(rows[0]),
            )
            digest_a = source_observation_sha256(rows[0])
            digest_b = source_observation_sha256(rows[0])
            self.assertEqual(digest_a, digest_b)

            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(SegmentError):
                load_raw_track_observations(empty)

            bad = root / "bad.jsonl"
            bad.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(SegmentError):
                load_raw_track_observations(bad)

            non_obj = root / "non_obj.jsonl"
            non_obj.write_text("[1,2]\n", encoding="utf-8")
            with self.assertRaises(SegmentError):
                load_raw_track_observations(non_obj)

            bad_track = root / "bad_track.jsonl"
            _write_jsonl(bad_track, [{**_obs(1, 0), "track_id": 0}])
            with self.assertRaises(SegmentError):
                load_raw_track_observations(bad_track)

            nan_bbox = root / "nan.jsonl"
            with nan_bbox.open("w", encoding="utf-8") as handle:
                handle.write(
                    '{"frame_index":0,"timestamp_sec":0.0,"track_id":1,'
                    '"class_id":0,"class_name":"person","confidence":0.9,'
                    '"bbox_xyxy":[0,0,10,NaN]}\n'
                )
            with self.assertRaises(SegmentError):
                load_raw_track_observations(nan_bbox)

            zero = root / "zero.jsonl"
            _write_jsonl(zero, [_obs(1, 0, bbox=[10, 10, 10, 20])])
            with self.assertRaises(SegmentError):
                load_raw_track_observations(zero)

            dup = root / "dup.jsonl"
            _write_jsonl(dup, [_obs(1, 0), _obs(1, 0)])
            with self.assertRaises(SegmentError):
                load_raw_track_observations(dup)


class SegmentApplicationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        # split track 10: frames 0,1,2,3(amb),5,6,7  gap 4 unused, gap configured 3-4
        # but 3 is ambiguous so not in gap as observation in gap - wait gap is 3,4
        # ambiguous 3 means frame 3 is observation AND ambiguous - cannot be in gap.
        # Use gap [4,4] and ambiguous [3].
        tracks = [
            _obs(10, 0),
            _obs(10, 1),
            _obs(10, 2),
            _obs(10, 3),
            _obs(10, 5),
            _obs(10, 6),
            _obs(10, 7),
            _obs(20, 0),
            _obs(20, 1),
            _obs(30, 10),
            _obs(30, 11),
        ]
        tracks_path = root / "tracks.jsonl"
        _write_jsonl(tracks_path, tracks)

        decisions = _decisions_doc(
            [
                _split_decision(
                    raw_track_id=10,
                    ambiguous=[3],
                    gaps=[[4, 4]],
                    segments=[
                        {
                            "segment_id": "raw_10_s01",
                            "raw_track_id": 10,
                            "frame_min": 0,
                            "frame_max": 2,
                            "include_existing_observations_only": True,
                            "proven_physical_identity": False,
                            "team_assignment": None,
                            "global_id": None,
                        },
                        {
                            "segment_id": "raw_10_s02",
                            "raw_track_id": 10,
                            "frame_min": 5,
                            "frame_max": 7,
                            "include_existing_observations_only": True,
                            "proven_physical_identity": False,
                            "team_assignment": None,
                            "global_id": None,
                        },
                    ],
                ),
                _control_decision(20),
            ]
        )
        decisions_path = root / "decisions.yaml"
        decisions_path.write_text(
            yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
        )
        policy_path = root / "policy.yaml"
        policy_path.write_text(
            DEFAULT_POLICY.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return {
            "tracks": tracks_path,
            "decisions": decisions_path,
            "policy": policy_path,
            "tracks_rows": tracks,
        }

    def test_application_and_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._fixture(root)
            out = root / "seg_out"
            hashes = {
                p: _sha256(p)
                for p in (paths["tracks"], paths["decisions"], paths["policy"])
            }
            result = run_build_manual_track_segment_view(
                tracks=paths["tracks"],
                segmentation_policy=paths["policy"],
                segment_decisions=paths["decisions"],
                output_dir=out,
            )
            self.assertEqual(result["raw_track_count"], 3)
            self.assertEqual(result["raw_observation_count"], 11)
            self.assertEqual(result["split_candidate_raw_track_count"], 1)
            self.assertEqual(result["no_split_control_raw_track_count"], 1)
            self.assertEqual(result["preserved_full_raw_track_count"], 1)
            self.assertEqual(result["manual_split_segment_count"], 2)
            self.assertEqual(result["total_segment_count"], 4)
            self.assertEqual(result["assigned_observation_count"], 10)
            self.assertEqual(result["unassigned_observation_count"], 1)
            self.assertEqual(result["created_observation_count"], 0)
            self.assertFalse(result["raw_tracks_mutated"])
            self.assertFalse(result["global_id_rewrite_performed"])
            self.assertFalse(result["reid_recomputation_performed"])

            names = sorted(p.name for p in out.iterdir() if p.is_file())
            self.assertEqual(
                names,
                sorted([SEGMENTS_NAME, SEGMENT_OBS_NAME, UNASSIGNED_NAME, SUMMARY_NAME]),
            )

            segments = [
                json.loads(line)
                for line in (out / SEGMENTS_NAME).read_text().splitlines()
                if line.strip()
            ]
            assigned = [
                json.loads(line)
                for line in (out / SEGMENT_OBS_NAME).read_text().splitlines()
                if line.strip()
            ]
            unassigned = [
                json.loads(line)
                for line in (out / UNASSIGNED_NAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue((out / SEGMENTS_NAME).read_text().endswith("\n"))
            self.assertEqual(len(unassigned), 1)
            self.assertEqual(unassigned[0]["frame_index"], 3)
            self.assertEqual(
                unassigned[0]["reason"], "manual_ambiguous_existing_observation"
            )
            self.assertNotIn("segment_id", unassigned[0])

            s01 = [r for r in assigned if r["segment_id"] == "raw_10_s01"]
            s02 = [r for r in assigned if r["segment_id"] == "raw_10_s02"]
            self.assertEqual([r["frame_index"] for r in s01], [0, 1, 2])
            self.assertEqual([r["frame_index"] for r in s02], [5, 6, 7])
            # no frame 4 observation created
            self.assertFalse(any(r["frame_index"] == 4 for r in assigned + unassigned))

            control = [s for s in segments if s["raw_track_id"] == 20][0]
            self.assertEqual(control["segment_kind"], "no_split_control")
            self.assertEqual(control["segment_id"], "raw_20_full")
            preserved = [s for s in segments if s["raw_track_id"] == 30][0]
            self.assertEqual(preserved["segment_kind"], "preserved_full_track")
            self.assertEqual(preserved["segment_id"], "raw_30_full")

            # source object preserved; no injected segment_id
            for row in assigned:
                nested = row["source_observation"]
                self.assertEqual(nested["track_id"], row["raw_track_id"])
                self.assertNotIn("segment_id", nested)
                self.assertEqual(
                    row["source_observation_sha256"],
                    source_observation_sha256(nested),
                )

            for path, digest in hashes.items():
                self.assertEqual(_sha256(path), digest)

            # collision
            with self.assertRaises(SegmentError):
                run_build_manual_track_segment_view(
                    tracks=paths["tracks"],
                    segmentation_policy=paths["policy"],
                    segment_decisions=paths["decisions"],
                    output_dir=out,
                )
            run_build_manual_track_segment_view(
                tracks=paths["tracks"],
                segmentation_policy=paths["policy"],
                segment_decisions=paths["decisions"],
                output_dir=out,
                overwrite=True,
            )
            self.assertEqual(
                list(out.parent.glob(f"_tmp_reid_segments_{out.name}_*")), []
            )

            # deterministic
            out2 = root / "seg_out2"
            run_build_manual_track_segment_view(
                tracks=paths["tracks"],
                segmentation_policy=paths["policy"],
                segment_decisions=paths["decisions"],
                output_dir=out2,
            )
            self.assertEqual(
                (out / SEGMENTS_NAME).read_text(),
                (out2 / SEGMENTS_NAME).read_text(),
            )

    def test_empty_segment_and_gap_and_uncovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "policy.yaml"
            policy_path.write_text(
                DEFAULT_POLICY.read_text(encoding="utf-8"), encoding="utf-8"
            )

            # empty segment: range with no observations
            tracks = [_obs(10, 0), _obs(10, 1), _obs(10, 5)]
            tracks_path = root / "tracks.jsonl"
            _write_jsonl(tracks_path, tracks)
            decisions = _decisions_doc(
                [
                    _split_decision(
                        raw_track_id=10,
                        gaps=[[2, 4]],
                        segments=[
                            {
                                "segment_id": "raw_10_s01",
                                "raw_track_id": 10,
                                "frame_min": 0,
                                "frame_max": 1,
                                "include_existing_observations_only": True,
                                "proven_physical_identity": False,
                                "team_assignment": None,
                                "global_id": None,
                            },
                            {
                                "segment_id": "raw_10_s02",
                                "raw_track_id": 10,
                                "frame_min": 8,
                                "frame_max": 9,
                                "include_existing_observations_only": True,
                                "proven_physical_identity": False,
                                "team_assignment": None,
                                "global_id": None,
                            },
                        ],
                    )
                ]
            )
            # uncovered frame 5
            decisions_path = root / "decisions.yaml"
            decisions_path.write_text(yaml.safe_dump(decisions), encoding="utf-8")
            with self.assertRaises(SegmentError):
                run_build_manual_track_segment_view(
                    tracks=tracks_path,
                    segmentation_policy=policy_path,
                    segment_decisions=decisions_path,
                    output_dir=root / "out_empty",
                )
            self.assertFalse((root / "out_empty").exists())

            # observation inside gap
            tracks2 = [_obs(10, 0), _obs(10, 1), _obs(10, 3), _obs(10, 5)]
            tracks_path2 = root / "tracks2.jsonl"
            _write_jsonl(tracks_path2, tracks2)
            decisions2 = _decisions_doc(
                [
                    _split_decision(
                        raw_track_id=10,
                        gaps=[[2, 4]],
                        ambiguous=[],
                        segments=[
                            {
                                "segment_id": "raw_10_s01",
                                "raw_track_id": 10,
                                "frame_min": 0,
                                "frame_max": 1,
                                "include_existing_observations_only": True,
                                "proven_physical_identity": False,
                                "team_assignment": None,
                                "global_id": None,
                            },
                            {
                                "segment_id": "raw_10_s02",
                                "raw_track_id": 10,
                                "frame_min": 5,
                                "frame_max": 5,
                                "include_existing_observations_only": True,
                                "proven_physical_identity": False,
                                "team_assignment": None,
                                "global_id": None,
                            },
                        ],
                    )
                ]
            )
            decisions_path2 = root / "decisions2.yaml"
            decisions_path2.write_text(yaml.safe_dump(decisions2), encoding="utf-8")
            with self.assertRaises(SegmentError):
                run_build_manual_track_segment_view(
                    tracks=tracks_path2,
                    segmentation_policy=policy_path,
                    segment_decisions=decisions_path2,
                    output_dir=root / "out_gap",
                )

            # unknown decision track
            decisions3 = _decisions_doc([_split_decision(raw_track_id=999)])
            decisions_path3 = root / "decisions3.yaml"
            decisions_path3.write_text(yaml.safe_dump(decisions3), encoding="utf-8")
            with self.assertRaises(SegmentError):
                run_build_manual_track_segment_view(
                    tracks=tracks_path,
                    segmentation_policy=policy_path,
                    segment_decisions=decisions_path3,
                    output_dir=root / "out_unknown",
                )

            # listed ambiguous missing from source
            decisions4 = _decisions_doc(
                [
                    _split_decision(
                        raw_track_id=10,
                        ambiguous=[9],
                        gaps=[[2, 4]],
                        segments=[
                            {
                                "segment_id": "raw_10_s01",
                                "raw_track_id": 10,
                                "frame_min": 0,
                                "frame_max": 1,
                                "include_existing_observations_only": True,
                                "proven_physical_identity": False,
                                "team_assignment": None,
                                "global_id": None,
                            },
                            {
                                "segment_id": "raw_10_s02",
                                "raw_track_id": 10,
                                "frame_min": 5,
                                "frame_max": 5,
                                "include_existing_observations_only": True,
                                "proven_physical_identity": False,
                                "team_assignment": None,
                                "global_id": None,
                            },
                        ],
                    )
                ]
            )
            # need tracks that cover segments without uncovered: 0,1,5 only - but ambiguous 9 missing
            tracks4 = [_obs(10, 0), _obs(10, 1), _obs(10, 5)]
            tracks_path4 = root / "tracks4.jsonl"
            _write_jsonl(tracks_path4, tracks4)
            decisions_path4 = root / "decisions4.yaml"
            decisions_path4.write_text(yaml.safe_dump(decisions4), encoding="utf-8")
            with self.assertRaises(SegmentError):
                run_build_manual_track_segment_view(
                    tracks=tracks_path4,
                    segmentation_policy=policy_path,
                    segment_decisions=decisions_path4,
                    output_dir=root / "out_amb",
                )


class ForbiddenImportTests(unittest.TestCase):
    def test_no_cv2_torch_in_segments_module(self) -> None:
        import football_analytics.reid.segments as segments_mod
        import inspect

        src = Path(segments_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import cv2", src)
        self.assertNotIn("import torch", src)
        self.assertNotIn("sklearn", src)
        self.assertNotIn("VideoCapture", src)
        self.assertNotIn('open("global_id_map', src)
        self.assertNotIn('Path("global_id_map', src)

        loader_src = "\n".join(
            [
                inspect.getsource(segments_mod.load_raw_track_observations),
                inspect.getsource(segments_mod.load_segmentation_policy),
                inspect.getsource(segments_mod.load_segment_decisions),
            ]
        )
        self.assertIn('path.open("r", encoding="utf-8")', loader_src)
        self.assertNotIn('.open("w"', loader_src)
        self.assertNotIn(".write_text(", loader_src)
        self.assertNotIn(".write_bytes(", loader_src)

    def test_help_has_expected_args_only(self) -> None:
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "build_manual_track_segment_view.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--tracks", proc.stdout)
        self.assertIn("--segmentation-policy", proc.stdout)
        self.assertIn("--segment-decisions", proc.stdout)
        self.assertIn("--output-dir", proc.stdout)
        self.assertNotIn("--video", proc.stdout)
        self.assertNotIn("--crop-manifest", proc.stdout)
        self.assertNotIn("--global-id", proc.stdout)
        self.assertNotIn("--checkpoint", proc.stdout)


if __name__ == "__main__":
    unittest.main()
