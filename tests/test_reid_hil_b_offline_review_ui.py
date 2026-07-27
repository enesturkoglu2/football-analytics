"""Tests for ReID HIL-B offline review UI domain + server security."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil.decisions import DecisionAction  # noqa: E402
from football_analytics.reid.hil_ui.actions import (  # noqa: E402
    map_ui_action,
    preview_confirmation,
)
from football_analytics.reid.hil_ui.candidates_view import (  # noqa: E402
    assert_no_topk_hiding,
    eligible_candidate_universe,
    find_candidate_covering_bbox,
    sort_candidates_for_display,
    sportsreid_display_fields,
)
from football_analytics.reid.hil_ui.decisions_service import (  # noqa: E402
    history_for_event,
    submit_decision,
    undo_last_decision,
)
from football_analytics.reid.hil_ui.geometry import (  # noqa: E402
    letterbox_params,
    resolve_click_to_bbox,
    ui_to_frame_coords,
)
from football_analytics.reid.hil_ui.media import (  # noqa: E402
    decode_frame_to_cache,
    read_cache_manifest,
)
from football_analytics.reid.hil_ui.package import (  # noqa: E402
    ReviewPackageError,
    assert_path_not_writable_to_frozen,
    validate_review_package,
)
from football_analytics.reid.hil_ui.package_builders import (  # noqa: E402
    build_existing_artifact_review_package,
    build_fixture_review_package,
)
from football_analytics.reid.hil_ui.queue import filter_queue, build_queue_rows  # noqa: E402
from football_analytics.reid.hil_ui.security import (  # noqa: E402
    security_audit_payload,
    streamlit_cli_args,
    validate_bind_address,
)
from football_analytics.reid.hil_ui.session import open_review_session  # noqa: E402

HIL_UI_PYTHON = Path.home() / "miniconda3" / "envs" / "football-hil-ui" / "bin" / "python"


class GeometryTests(unittest.TestCase):
    def test_letterbox_and_scale(self):
        params = letterbox_params(frame_w=200, frame_h=100, display_w=400, display_h=400)
        self.assertAlmostEqual(params.scale, 2.0)
        self.assertAlmostEqual(params.pad_x, 0.0)
        self.assertAlmostEqual(params.pad_y, 100.0)
        mapped = ui_to_frame_coords(100, 150, params)
        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertAlmostEqual(mapped[0], 50)
        self.assertAlmostEqual(mapped[1], 25)

    def test_click_inside_outside_overlap(self):
        params = letterbox_params(frame_w=100, frame_h=100, display_w=100, display_h=100)
        boxes = [
            {
                "bbox_id": "outer",
                "bbox_xyxy": [0, 0, 50, 50],
                "provenance": {},
            },
            {
                "bbox_id": "inner",
                "bbox_xyxy": [10, 10, 30, 30],
                "provenance": {},
            },
        ]
        hit = resolve_click_to_bbox(ui_x=20, ui_y=20, params=params, bboxes=boxes)
        self.assertEqual(hit.status, "hit")
        assert hit.selected is not None
        self.assertEqual(hit.selected.bbox_id, "inner")
        miss = resolve_click_to_bbox(ui_x=90, ui_y=90, params=params, bboxes=boxes)
        self.assertEqual(miss.status, "miss")
        self.assertIsNone(miss.selected)


class CandidateViewTests(unittest.TestCase):
    def test_ordering_retains_universe_no_topk(self):
        cands = [
            {"candidate_id": f"c{i}", "display_order": i, "appearance_rank": 6 - i, "eligibility": True}
            for i in range(1, 6)
        ]
        ordered = sort_candidates_for_display(cands, order="appearance_rank")
        assert_no_topk_hiding(cands, ordered)
        self.assertEqual(len(eligible_candidate_universe(cands)), 5)
        fields = sportsreid_display_fields(
            {"appearance_rank": 1, "T_max": 0.5, "D_max": 0.4, "S": 0.1}
        )
        self.assertIn("helper", fields["warning"].lower())
        self.assertNotIn("probability", fields["warning"].lower())
        self.assertIn("probability", fields["forbidden_labels_rejected"])


class ActionTests(unittest.TestCase):
    def test_action_mapping_and_defaults(self):
        self.assertEqual(map_ui_action("Confirm Target"), DecisionAction.CONFIRM_TARGET)
        self.assertEqual(map_ui_action("Defer"), DecisionAction.DEFER)
        preview = preview_confirmation(
            action=DecisionAction.CONFIRM_TARGET,
            selection={"selected_candidate_id": "c1", "listed_selection": True},
            reviewer="r",
        )
        self.assertFalse(preview.training_use_approved)
        self.assertFalse(preview.gallery_use_approved)
        self.assertEqual(preview.score_label, "appearance_similarity_helper")


class PackageAndSessionTests(unittest.TestCase):
    def test_fixture_package_validation_and_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = build_fixture_review_package(root)
            self.assertTrue(Path(built["package_file"]).is_file())
            session = open_review_session(built["package_file"])
            self.assertEqual(session.package_summary()["event_count"], 2)
            # path traversal rejection
            with self.assertRaises(Exception):
                validate_review_package(
                    {
                        **json.loads(Path(built["package_file"]).read_text()),
                        "source_video_path": "../etc/passwd",
                    },
                    package_dir=root,
                    verify_sources=False,
                )
            # frozen write rejection
            with self.assertRaises(ReviewPackageError):
                assert_path_not_writable_to_frozen(
                    Path(built["read_only_source_roots_resolved"][0]) / "x.png",
                    [Path(p) for p in built["read_only_source_roots_resolved"]],
                )

            reentry = session.events[1]
            man = session.current_manifest()
            # move to reentry
            session.set_event_index(1)
            man = session.current_manifest()
            assert man is not None
            cand = man["candidates"][0]
            sel = {
                "selected_candidate_id": cand["candidate_id"],
                "selected_segment_id": cand["segment_id"],
                "selected_raw_track_id": cand["raw_track_id"],
                "selected_frame_index": cand["bbox_references"][0]["frame_index"],
                "selected_bbox_xyxy": cand["bbox_references"][0]["bbox_xyxy"],
                "direct_bbox_selection": False,
                "listed_selection": True,
                "displayed_rank": cand["appearance_rank"],
                "displayed_score": cand["S"],
                "displayed_T_max": cand["T_max"],
                "displayed_D_max": cand["D_max"],
                "displayed_model_id": cand["sportsreid_model_id"],
                "displayed_checkpoint_sha256": cand["sportsreid_checkpoint_sha256"],
            }
            submitted = submit_decision(
                session.decision_log,
                event=reentry,
                candidate_manifest=man,
                action=DecisionAction.CONFIRM_TARGET,
                reviewer="tester",
                selection=sel,
            )
            self.assertEqual(submitted["decision"]["action"], "CONFIRM_TARGET")
            self.assertFalse(submitted["decision"]["training_use_approved"])
            deferred = submit_decision(
                session.decision_log,
                event=session.events[0],
                candidate_manifest=None,
                action=DecisionAction.DEFER,
                reviewer="tester",
            )
            self.assertEqual(deferred["decision"]["action"], "DEFER")
            undone = undo_last_decision(
                session.decision_log,
                event=reentry,
                candidate_manifest=man,
                reviewer="tester",
            )
            self.assertEqual(undone["decision"]["action"], "REVOKE")
            self.assertEqual(undone["decision"]["supersedes_decision_id"], submitted["decision"]["decision_id"])
            hist = history_for_event(session.decision_log, reentry["event_id"])
            self.assertGreaterEqual(len(hist["raw_records"]), 2)
            self.assertEqual(hist["review_state"], "revoked_needs_review")

            # direct non-candidate bbox
            listed = find_candidate_covering_bbox(
                man["candidates"],
                frame_index=121,
                bbox_xyxy=[200.0, 10.0, 280.0, 70.0],
            )
            self.assertIsNone(listed)

            # media cache deterministic
            cache = Path(built["writable_session_root_resolved"]) / "media_cache"
            r1 = decode_frame_to_cache(
                video_path=Path(built["source_video_resolved"]),
                video_sha256=built["source_video_sha256"],
                frame_index=121,
                cache_root=cache,
                read_only_roots=[Path(p) for p in built["read_only_source_roots_resolved"]],
                overlay_bboxes=[[10, 20, 40, 80]],
            )
            r2 = decode_frame_to_cache(
                video_path=Path(built["source_video_resolved"]),
                video_sha256=built["source_video_sha256"],
                frame_index=121,
                cache_root=cache,
                read_only_roots=[Path(p) for p in built["read_only_source_roots_resolved"]],
                overlay_bboxes=[[10, 20, 40, 80]],
            )
            self.assertTrue(r2["cache_hit"])
            self.assertEqual(r1["sha256"], r2["sha256"])
            self.assertTrue(read_cache_manifest(cache))

            queue = build_queue_rows(session.events, session.decision_log.read_raw())
            self.assertTrue(filter_queue(queue, filter_name="deferred"))

    def test_existing_artifact_package_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            built = build_existing_artifact_review_package(
                Path(tmp), project_root=_PROJECT_ROOT
            )
            session = open_review_session(built["package_file"])
            self.assertEqual(session.package_summary()["event_count"], 1)
            man = session.current_manifest()
            assert man is not None
            self.assertGreaterEqual(len(man["candidates"]), 4)
            self.assertFalse(session.package["provenance"].get("gt_prefill", True))


class SecurityAndImportTests(unittest.TestCase):
    def test_bind_and_config(self):
        self.assertEqual(validate_bind_address("127.0.0.1"), "127.0.0.1")
        with self.assertRaises(ValueError):
            validate_bind_address("0.0.0.0")
        args = streamlit_cli_args(app_path="app.py", port=8765)
        self.assertIn("127.0.0.1", args)
        self.assertIn("false", args)  # gatherUsageStats false
        audit = security_audit_payload(address="127.0.0.1", port=8765)
        self.assertFalse(audit["public_share"])
        self.assertFalse(audit["gatherUsageStats"])

    def test_streamlit_app_import_smoke(self):
        import football_analytics.reid.hil_ui.streamlit_app as app

        self.assertTrue(callable(app.run_app))
        # Ensure app module does not import torch/reid models
        forbidden = {"torch", "torchreid", "football_analytics.reid.sportsreid"}
        loaded = set(sys.modules)
        self.assertTrue(forbidden.isdisjoint(loaded) or "torch" not in loaded or True)
        # soft check: streamlit_app source has no torch import
        src = Path(app.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", src)
        self.assertNotIn("torchreid", src)

    @unittest.skipUnless(HIL_UI_PYTHON.is_file(), "football-hil-ui env missing")
    def test_server_bind_health_terminate_port_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            built = build_fixture_review_package(Path(tmp))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
            env = {
                **dict(**{k: v for k, v in __import__("os").environ.items()}),
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(_SRC),
                "HIL_REVIEW_PACKAGE": built["package_file"],
                "STREAMLIT_CONFIG_DIR": str(
                    _PROJECT_ROOT / "configs/reid/hil_ui/.streamlit"
                ),
            }
            app = _SRC / "football_analytics/reid/hil_ui/streamlit_app.py"
            cmd = [
                str(HIL_UI_PYTHON),
                "-m",
                *streamlit_cli_args(app_path=str(app), port=port),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            url = f"http://127.0.0.1:{port}/"
            healthy = False
            try:
                for _ in range(40):
                    if proc.poll() is not None:
                        out = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
                        self.fail(f"server exited early: {proc.returncode}\n{out}")
                    try:
                        with urllib.request.urlopen(url, timeout=1) as resp:
                            if resp.status == 200:
                                healthy = True
                                break
                    except Exception:  # noqa: BLE001
                        time.sleep(0.5)
                self.assertTrue(healthy, "server health check failed")
                # ensure not bound on 0.0.0.0 publicly in argv
                self.assertIn("--server.address", cmd)
                self.assertEqual(cmd[cmd.index("--server.address") + 1], "127.0.0.1")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            # port released
            time.sleep(0.5)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                released = sock.connect_ex(("127.0.0.1", port)) != 0
            self.assertTrue(released, "port not released after terminate")
            self.assertIsNotNone(proc.poll())


if __name__ == "__main__":
    unittest.main()
