"""Unit tests for R2C2 safe SportsReID adapter, registry, and loader contracts."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.embedding import (  # noqa: E402
    EMBEDDING_DIM,
    MODEL_ID_MARKET1501,
    MODEL_ID_SPORTSREID_SOCCERNET,
    EmbeddingError,
    load_reid_osnet_by_model_id,
)
from football_analytics.reid.model_registry import (  # noqa: E402
    ModelRegistryError,
    get_reid_model_spec,
    list_reid_model_ids,
    load_model_registry,
)
from football_analytics.reid.safe_checkpoint import (  # noqa: E402
    SafeCheckpointError,
    apply_sanitized_state_dict_to_osnet,
    load_sanitized_state_dict_only,
    reject_non_sanitized_sportsreid_checkpoint,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class _FakeModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)
        self.classifier = torch.nn.Linear(4, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


class RegistryTests(unittest.TestCase):
    def test_registry_lists_both_model_ids(self) -> None:
        ids = list_reid_model_ids()
        self.assertIn(MODEL_ID_MARKET1501, ids)
        self.assertIn(MODEL_ID_SPORTSREID_SOCCERNET, ids)

    def test_explicit_lookup_and_unknown_no_fallback(self) -> None:
        spec = get_reid_model_spec(MODEL_ID_SPORTSREID_SOCCERNET)
        self.assertEqual(spec["model_id"], MODEL_ID_SPORTSREID_SOCCERNET)
        self.assertTrue(spec["weights_only"])
        self.assertTrue(spec["safe_load_required"])
        with self.assertRaises(ModelRegistryError) as ctx:
            get_reid_model_spec("does_not_exist")
        self.assertIn("no fallback executed", str(ctx.exception))

    def test_market1501_entry_unchanged_contract(self) -> None:
        spec = get_reid_model_spec(MODEL_ID_MARKET1501)
        self.assertEqual(
            spec["sha256"],
            "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154",
        )
        self.assertFalse(spec["safe_load_required"])


class SafeLoaderContractTests(unittest.TestCase):
    def test_weights_only_true_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sd.pth"
            sd = OrderedDict({"backbone.weight": torch.randn(4, 4)})
            torch.save(sd, path)
            digest = _sha256(path)
            with mock.patch("torch.load", wraps=torch.load) as mocked:
                loaded = load_sanitized_state_dict_only(path, expected_sha256=digest)
            mocked.assert_called()
            kwargs = mocked.call_args.kwargs
            self.assertTrue(kwargs.get("weights_only"))
            self.assertEqual(kwargs.get("map_location"), "cpu")
            self.assertEqual(set(loaded.keys()), {"backbone.weight"})

    def test_sha_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sd.pth"
            torch.save(OrderedDict({"a": torch.ones(2)}), path)
            with self.assertRaises(SafeCheckpointError):
                load_sanitized_state_dict_only(
                    path, expected_sha256="0" * 64
                )

    def test_full_training_checkpoint_path_rejected(self) -> None:
        banned = Path(
            "/home/enesturkoglu2/projects/soccernet/checkpoints/"
            "_quarantine_sportsreid_osnet_20260727T205544_17463/osnet_checkpoint.pth.tar"
        )
        if banned.is_file():
            with self.assertRaises(SafeCheckpointError) as ctx:
                reject_non_sanitized_sportsreid_checkpoint(banned.resolve())
            self.assertIn("non-sanitized", str(ctx.exception).lower())

    def test_non_dict_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pth"
            torch.save(torch.ones(3), path)
            digest = _sha256(path)
            with self.assertRaises(SafeCheckpointError):
                load_sanitized_state_dict_only(path, expected_sha256=digest)

    def test_optimizer_wrapper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.pth"
            payload = {
                "state_dict": OrderedDict({"backbone.weight": torch.randn(4, 4)}),
                "optimizer": {"param_groups": []},
            }
            torch.save(payload, path)
            digest = _sha256(path)
            with self.assertRaises(SafeCheckpointError):
                load_sanitized_state_dict_only(path, expected_sha256=digest)

    def test_non_tensor_value_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pth"
            # weights_only may itself reject plain python objects; either outcome is fail-closed.
            try:
                torch.save(OrderedDict({"a": "not-a-tensor"}), path)
            except Exception:
                self.skipTest("torch.save refused non-tensor payload in this build")
            digest = _sha256(path)
            with self.assertRaises(SafeCheckpointError):
                load_sanitized_state_dict_only(path, expected_sha256=digest)

    def test_classifier_only_mismatch_ignored_backbone_ok(self) -> None:
        model = _FakeModule()
        state = OrderedDict(
            {
                "backbone.weight": model.backbone.weight.detach().clone(),
                "classifier.weight": torch.randn(99, 4),
                "classifier.bias": torch.randn(99),
            }
        )
        coverage = apply_sanitized_state_dict_to_osnet(model, state)
        self.assertEqual(
            coverage["decision"], "SPORTSREID_OSNET_FEATURE_EXTRACTOR_COMPATIBLE"
        )
        self.assertEqual(coverage["backbone_coverage"], 1.0)
        self.assertTrue(coverage["ignored_classifier_keys"])

    def test_backbone_mismatch_rejected(self) -> None:
        model = _FakeModule()
        state = OrderedDict({"backbone.weight": torch.randn(3, 3)})
        with self.assertRaises(SafeCheckpointError):
            apply_sanitized_state_dict_to_osnet(model, state)


class ModelSelectionTests(unittest.TestCase):
    def test_unknown_model_id_no_silent_fallback(self) -> None:
        with self.assertRaises(EmbeddingError) as ctx:
            load_reid_osnet_by_model_id("not_a_real_model")
        self.assertIn("no fallback executed", str(ctx.exception))

    def test_r2b_module_not_imported_by_safe_checkpoint(self) -> None:
        import football_analytics.reid.safe_checkpoint as sc

        self.assertFalse(hasattr(sc, "multiframe_r2b"))
        # Safe checkpoint must not depend on R2B helpers (inspect module source).
        source = Path(sc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("multiframe_r2b", source)
        self.assertNotIn("run_reid_r2b_market1501_multiframe", source)


class LiveSanitizedAssetTests(unittest.TestCase):
    """Optional live checks against the immutable sanitized SportsReID asset."""

    @classmethod
    def setUpClass(cls) -> None:
        spec = get_reid_model_spec(MODEL_ID_SPORTSREID_SOCCERNET)
        cls.spec = spec
        cls.path = Path(spec["checkpoint_path"])
        cls.available = cls.path.is_file()

    def test_sanitized_state_dict_safe_load(self) -> None:
        if not self.available:
            self.skipTest("sanitized SportsReID checkpoint not present")
        state = load_sanitized_state_dict_only(
            self.path, expected_sha256=str(self.spec["sha256"])
        )
        self.assertTrue(state)
        self.assertTrue(all(torch.is_tensor(v) for v in state.values()))
        self.assertNotIn("optimizer", state)

    def test_expected_sha_success(self) -> None:
        if not self.available:
            self.skipTest("sanitized SportsReID checkpoint not present")
        actual = _sha256(self.path)
        self.assertEqual(actual, str(self.spec["sha256"]).lower())
        self.assertEqual(self.path.stat().st_size, int(self.spec["size_bytes"]))


if __name__ == "__main__":
    unittest.main()
