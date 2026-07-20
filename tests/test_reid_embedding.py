"""Unit tests for ReID crop embedding (no real checkpoint / sn-reid / crops)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.embedding import (
    EMBEDDING_DIM,
    INDEX_SCHEMA_VERSION,
    MODEL_NAME,
    NPZ_NAME,
    SUMMARY_SCHEMA_VERSION,
    EmbeddingError,
    build_osnet_cpu_model,
    embed_tensors,
    l2_normalize_rows,
    load_and_preprocess_crop,
    load_crop_manifest_rows,
    load_osnet_checkpoint_weights,
    preprocess_crop_bgr,
    resolve_crop_path,
    run_extract_reid_embeddings,
    verify_checkpoint,
    verify_sn_reid_root,
)
from football_analytics.reid.schema import build_crop_manifest_row
from football_analytics.reid.writers import ReIDWritersError, write_manifest_jsonl


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jpeg(path: Path, *, width: int = 40, height: int = 80, color=(0, 0, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed writing jpeg {path}")


def _init_sn_reid_repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "torchreid").mkdir(parents=True, exist_ok=True)
    init_py = root / "torchreid" / "__init__.py"
    if not init_py.exists():
        init_py.write_text("# test stub\n", encoding="utf-8")
    git_dir = root / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=test",
                "commit",
                "-m",
                "init",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return head


class DeterministicModel:
    """CPU mock returning (B, 512) float32 features derived from the batch."""

    def eval(self):
        return self

    def cpu(self):
        return self

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 4:
            raise AssertionError(batch.shape)
        b = batch.shape[0]
        base = torch.arange(EMBEDDING_DIM, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
        # Preserve order: each row gets a unique offset from the batch content.
        offsets = batch.reshape(b, -1).mean(dim=1, keepdim=True)
        return base + offsets + 1.0


class BadShapeModel(DeterministicModel):
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.ones((batch.shape[0], 128), dtype=torch.float32)


class ZeroNormModel(DeterministicModel):
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros((batch.shape[0], EMBEDDING_DIM), dtype=torch.float32)


class NanModel(DeterministicModel):
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        out = torch.ones((batch.shape[0], EMBEDDING_DIM), dtype=torch.float32)
        out[:, 0] = float("nan")
        return out


def _make_crop_bundle(tmp: Path, *, n: int = 3) -> tuple[Path, list[dict]]:
    crop_dir = tmp / "crops_out"
    crops = crop_dir / "crops" / "track_1"
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rank = i + 1
        frame = 10 + i * 10
        rel = f"crops/track_1/crop_{frame}_{rank}.jpg"
        _write_jpeg(
            crop_dir / rel,
            width=30 + i,
            height=60 + i,
            color=(i * 40, 80, 200 - i * 20),
        )
        row = build_crop_manifest_row(
            track_id=1,
            frame_index=frame,
            timestamp_sec=float(frame) / 25.0,
            source_video="synthetic.mp4",
            bbox_xyxy=[0, 0, 30 + i, 60 + i],
            detection_confidence=0.9,
            quality_score=1000.0 + i,
            selection_rank=rank,
        )
        rows.append(row)
    manifest = crop_dir / "crop_manifest.jsonl"
    write_manifest_jsonl(manifest, rows)
    return manifest, rows


class ManifestInputTests(unittest.TestCase):
    def test_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, rows = _make_crop_bundle(Path(tmp), n=2)
            loaded = load_crop_manifest_rows(manifest)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["crop_id"], rows[0]["crop_id"])

    def test_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crop_manifest.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(EmbeddingError):
                load_crop_manifest_rows(path)

    def test_bad_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crop_manifest.jsonl"
            path.write_text("{not-json\n", encoding="utf-8")
            with self.assertRaises(EmbeddingError):
                load_crop_manifest_rows(path)

    def test_duplicate_crop_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jpeg(root / "crops/a.jpg")
            _write_jpeg(root / "crops/b.jpg")
            row = build_crop_manifest_row(
                track_id=1,
                frame_index=1,
                timestamp_sec=0.0,
                source_video="v.mp4",
                bbox_xyxy=[0, 0, 10, 20],
                detection_confidence=0.8,
                quality_score=10.0,
                selection_rank=1,
            )
            row_a = dict(row)
            row_a["crop_relative_path"] = "crops/a.jpg"
            row_b = dict(row)
            row_b["crop_id"] = row_a["crop_id"]
            row_b["crop_relative_path"] = "crops/b.jpg"
            row_b["frame_index"] = 2
            manifest = root / "crop_manifest.jsonl"
            write_manifest_jsonl(manifest, [row_a, row_b])
            with self.assertRaises(EmbeddingError):
                load_crop_manifest_rows(manifest)

    def test_duplicate_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jpeg(root / "crops/a.jpg")
            row1 = build_crop_manifest_row(
                track_id=1,
                frame_index=1,
                timestamp_sec=0.0,
                source_video="v.mp4",
                bbox_xyxy=[0, 0, 10, 20],
                detection_confidence=0.8,
                quality_score=10.0,
                selection_rank=1,
            )
            row2 = build_crop_manifest_row(
                track_id=1,
                frame_index=2,
                timestamp_sec=0.1,
                source_video="v.mp4",
                bbox_xyxy=[0, 0, 10, 20],
                detection_confidence=0.8,
                quality_score=10.0,
                selection_rank=2,
            )
            row1["crop_relative_path"] = "crops/a.jpg"
            row2["crop_relative_path"] = "crops/a.jpg"
            manifest = root / "crop_manifest.jsonl"
            write_manifest_jsonl(manifest, [row1, row2])
            with self.assertRaises(EmbeddingError):
                load_crop_manifest_rows(manifest)

    def test_missing_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_crop_manifest_row(
                track_id=1,
                frame_index=1,
                timestamp_sec=0.0,
                source_video="v.mp4",
                bbox_xyxy=[0, 0, 10, 20],
                detection_confidence=0.8,
                quality_score=10.0,
                selection_rank=1,
            )
            manifest = root / "crop_manifest.jsonl"
            write_manifest_jsonl(manifest, [row])
            with self.assertRaises(EmbeddingError):
                load_crop_manifest_rows(manifest)

    def test_absolute_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jpeg = root / "crop.jpg"
            _write_jpeg(jpeg)
            with self.assertRaises(EmbeddingError):
                resolve_crop_path(root, str(jpeg.resolve()))

    def test_parent_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "manifest_dir"
            root.mkdir()
            outside = Path(tmp) / "secret.jpg"
            _write_jpeg(outside)
            with self.assertRaises(EmbeddingError):
                resolve_crop_path(root, "../secret.jpg")

    def test_corrupt_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "crops" / "bad.jpg"
            bad.parent.mkdir(parents=True)
            bad.write_bytes(b"not-a-jpeg")
            row = build_crop_manifest_row(
                track_id=1,
                frame_index=1,
                timestamp_sec=0.0,
                source_video="v.mp4",
                bbox_xyxy=[0, 0, 10, 20],
                detection_confidence=0.8,
                quality_score=10.0,
                selection_rank=1,
            )
            row["crop_relative_path"] = "crops/bad.jpg"
            manifest = root / "crop_manifest.jsonl"
            write_manifest_jsonl(manifest, [row])
            with self.assertRaises(EmbeddingError):
                load_crop_manifest_rows(manifest)


class CheckpointTests(unittest.TestCase):
    def test_matching_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pth.tar"
            payload = b"fake-checkpoint-bytes"
            path.write_bytes(payload)
            info = verify_checkpoint(path, expected_sha256=_sha256_bytes(payload))
            self.assertEqual(info["sha256"], _sha256_bytes(payload))

    def test_mismatched_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pth.tar"
            path.write_bytes(b"fake-checkpoint-bytes")
            wrong = "0" * 64
            with self.assertRaises(EmbeddingError):
                verify_checkpoint(path, expected_sha256=wrong)

    def test_missing_checkpoint(self) -> None:
        with self.assertRaises(EmbeddingError):
            verify_checkpoint("/tmp/does-not-exist-reid-ckpt.pth.tar", expected_sha256="a" * 64)

    def test_sha_mismatch_skips_model_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest, _ = _make_crop_bundle(tmp_path, n=1)
            ckpt = tmp_path / "ckpt.pth.tar"
            ckpt.write_bytes(b"bytes")
            sn_root = tmp_path / "sn-reid"
            commit = _init_sn_reid_repo(sn_root)
            builder = mock.Mock(side_effect=AssertionError("model builder must not run"))
            loader = mock.Mock(side_effect=AssertionError("weight loader must not run"))
            with self.assertRaises(EmbeddingError):
                run_extract_reid_embeddings(
                    crop_manifest=manifest,
                    checkpoint=ckpt,
                    checkpoint_sha256="b" * 64,
                    sn_reid_root=sn_root,
                    expected_sn_reid_commit=commit,
                    output_dir=tmp_path / "emb_out",
                    model_builder=builder,
                    weight_loader=loader,
                )
            builder.assert_not_called()
            loader.assert_not_called()
            self.assertFalse((tmp_path / "emb_out").exists())


class SnReidRootTests(unittest.TestCase):
    def test_missing_root(self) -> None:
        with self.assertRaises(EmbeddingError):
            verify_sn_reid_root("/tmp/missing-sn-reid-root", expected_commit="abc")

    def test_missing_torchreid_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sn-reid"
            root.mkdir()
            with self.assertRaises(EmbeddingError):
                verify_sn_reid_root(root, expected_commit="abc")

    def test_commit_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sn-reid"
            _init_sn_reid_repo(root)
            with self.assertRaises(EmbeddingError):
                verify_sn_reid_root(root, expected_commit="0" * 40)

    def test_commit_mismatch_skips_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest, _ = _make_crop_bundle(tmp_path, n=1)
            ckpt = tmp_path / "ckpt.pth.tar"
            payload = b"ckpt"
            ckpt.write_bytes(payload)
            sn_root = tmp_path / "sn-reid"
            _init_sn_reid_repo(sn_root)
            builder = mock.Mock(side_effect=AssertionError("must not build"))
            with self.assertRaises(EmbeddingError):
                run_extract_reid_embeddings(
                    crop_manifest=manifest,
                    checkpoint=ckpt,
                    checkpoint_sha256=_sha256_bytes(payload),
                    sn_reid_root=sn_root,
                    expected_sn_reid_commit="deadbeef" * 5,
                    output_dir=tmp_path / "emb_out",
                    model_builder=builder,
                    weight_loader=mock.Mock(),
                )
            builder.assert_not_called()


class PreprocessTests(unittest.TestCase):
    def test_synthetic_shape_dtype_finite(self) -> None:
        image = np.zeros((50, 25, 3), dtype=np.uint8)
        image[:, :, 0] = 10
        image[:, :, 1] = 20
        image[:, :, 2] = 30
        tensor = preprocess_crop_bgr(image)
        self.assertEqual(tuple(tensor.shape), (3, 256, 128))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(tensor.device.type, "cpu")

    def test_bgr_to_rgb_behavior(self) -> None:
        blue = np.zeros((40, 20, 3), dtype=np.uint8)
        blue[:, :, 0] = 255  # BGR blue
        red = np.zeros((40, 20, 3), dtype=np.uint8)
        red[:, :, 2] = 255  # BGR red
        t_blue = preprocess_crop_bgr(blue)
        t_red = preprocess_crop_bgr(red)
        # After BGR→RGB, pure-blue and pure-red tensors must differ.
        self.assertFalse(torch.allclose(t_blue, t_red))
        # Channel-0 (R after conversion) should be higher for the red BGR image.
        self.assertGreater(float(t_red[0].mean()), float(t_blue[0].mean()))

    def test_corrupt_image_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jpg"
            path.write_bytes(b"nope")
            with self.assertRaises(EmbeddingError):
                load_and_preprocess_crop(path)

    def test_empty_array_errors(self) -> None:
        with self.assertRaises(EmbeddingError):
            preprocess_crop_bgr(np.zeros((0, 10, 3), dtype=np.uint8))


class EmbeddingMathTests(unittest.TestCase):
    def test_l2_normalize(self) -> None:
        raw = np.ones((2, EMBEDDING_DIM), dtype=np.float32) * np.array(
            [[2.0], [4.0]], dtype=np.float32
        )
        normed = l2_normalize_rows(raw)
        norms = np.linalg.norm(normed, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))

    def test_zero_norm_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            l2_normalize_rows(np.zeros((1, EMBEDDING_DIM), dtype=np.float32))

    def test_nan_rejected(self) -> None:
        arr = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
        arr[0, 0] = np.nan
        with self.assertRaises(EmbeddingError):
            l2_normalize_rows(arr)

    def test_wrong_dim_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            l2_normalize_rows(np.ones((2, 128), dtype=np.float32))

    def test_batching_order_and_remainder(self) -> None:
        model = DeterministicModel()
        tensors = [
            torch.full((3, 256, 128), float(i), dtype=torch.float32) for i in range(5)
        ]
        out = embed_tensors(model, tensors, batch_size=2)
        self.assertEqual(out.shape, (5, EMBEDDING_DIM))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5))
        # Distinct inputs → distinct embeddings after normalize.
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertFalse(np.allclose(out[i], out[j]))

    def test_bad_model_shape(self) -> None:
        tensors = [torch.zeros((3, 256, 128), dtype=torch.float32)]
        with self.assertRaises(EmbeddingError):
            embed_tensors(BadShapeModel(), tensors, batch_size=1)

    def test_zero_norm_model(self) -> None:
        tensors = [torch.ones((3, 256, 128), dtype=torch.float32)]
        with self.assertRaises(EmbeddingError):
            embed_tensors(ZeroNormModel(), tensors, batch_size=1)

    def test_nan_model(self) -> None:
        tensors = [torch.ones((3, 256, 128), dtype=torch.float32)]
        with self.assertRaises(EmbeddingError):
            embed_tensors(NanModel(), tensors, batch_size=1)

    def test_deterministic_repeat(self) -> None:
        model = DeterministicModel()
        tensors = [torch.full((3, 256, 128), 3.0, dtype=torch.float32) for _ in range(3)]
        a = embed_tensors(model, tensors, batch_size=2)
        b = embed_tensors(model, tensors, batch_size=2)
        self.assertTrue(np.array_equal(a, b))


class OutputPipelineTests(unittest.TestCase):
    def _run(self, tmp: Path, *, n: int = 5, batch_size: int = 2, overwrite: bool = False):
        manifest, rows = _make_crop_bundle(tmp, n=n)
        ckpt = tmp / "ckpt.pth.tar"
        payload = b"unit-test-checkpoint"
        ckpt.write_bytes(payload)
        sn_root = tmp / "sn-reid"
        commit = _init_sn_reid_repo(sn_root)
        out = tmp / "emb_out"

        def builder(*, model_name: str = MODEL_NAME):
            self.assertEqual(model_name, MODEL_NAME)
            return DeterministicModel()

        def loader(model, path):
            self.assertTrue(Path(path).is_file())
            self.assertIsInstance(model, DeterministicModel)

        result = run_extract_reid_embeddings(
            crop_manifest=manifest,
            checkpoint=ckpt,
            checkpoint_sha256=_sha256_bytes(payload),
            sn_reid_root=sn_root,
            expected_sn_reid_commit=commit,
            output_dir=out,
            batch_size=batch_size,
            overwrite=overwrite,
            model_builder=builder,
            weight_loader=loader,
        )
        return result, rows, out

    def test_npz_index_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, rows, out = self._run(Path(tmp), n=5, batch_size=2)
            self.assertEqual(result["crop_count"], 5)
            self.assertEqual(result["batch_count"], 3)
            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(
                names,
                ["crop_embeddings.npz", "crop_embeddings_index.jsonl", "embedding_summary.json"],
            )

            data = np.load(out / NPZ_NAME, allow_pickle=False)
            self.assertEqual(data["vectors"].shape, (5, EMBEDDING_DIM))
            self.assertEqual(data["vectors"].dtype, np.float32)
            self.assertNotEqual(data["vectors"].dtype, object)
            self.assertEqual(data["crop_ids"].dtype.kind, "U")
            self.assertEqual(list(data["track_ids"]), [1] * 5)
            self.assertEqual(list(data["frame_indices"]), [r["frame_index"] for r in rows])

            index_lines = (out / "crop_embeddings_index.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(index_lines), 5)
            embedding_rows = []
            for i, line in enumerate(index_lines):
                rec = json.loads(line)
                self.assertEqual(rec["embedding_row"], i)
                self.assertEqual(rec["crop_id"], rows[i]["crop_id"])
                self.assertEqual(rec["schema_version"], INDEX_SCHEMA_VERSION)
                self.assertAlmostEqual(rec["l2_norm"], 1.0, places=5)
                embedding_rows.append(rec["embedding_row"])
            self.assertEqual(embedding_rows, list(range(5)))

            summary = json.loads((out / "embedding_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["schema_version"], SUMMARY_SCHEMA_VERSION)
            self.assertEqual(summary["embeddings_shape"], [5, 512])
            self.assertFalse(summary["checkpoint_soccerNet_trained"])
            self.assertFalse(summary["pretrained_flag"])
            self.assertFalse(summary["feature_extractor_used"])
            self.assertFalse(summary["automatic_download_occurred"])
            self.assertEqual(summary["device"], "cpu")

    def test_collision_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first, rows, out = self._run(tmp_path, n=1, batch_size=1)
            manifest = out.parent / "crops_out" / "crop_manifest.jsonl"
            # Reuse the already-written inputs; only the output collision matters.
            ckpt = tmp_path / "ckpt.pth.tar"
            sn_root = tmp_path / "sn-reid"
            commit = _init_sn_reid_repo(sn_root)
            with self.assertRaises(ReIDWritersError):
                run_extract_reid_embeddings(
                    crop_manifest=manifest,
                    checkpoint=ckpt,
                    checkpoint_sha256=_sha256_bytes(b"unit-test-checkpoint"),
                    sn_reid_root=sn_root,
                    expected_sn_reid_commit=commit,
                    output_dir=out,
                    batch_size=1,
                    overwrite=False,
                    model_builder=lambda **_: DeterministicModel(),
                    weight_loader=lambda m, p: None,
                )
            self.assertEqual(first["status"], "ok")
            self.assertEqual(len(rows), 1)

    def test_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first, _, out = self._run(tmp_path, n=2, batch_size=1)
            manifest = tmp_path / "crops_out" / "crop_manifest.jsonl"
            ckpt = tmp_path / "ckpt.pth.tar"
            sn_root = tmp_path / "sn-reid"
            commit = _init_sn_reid_repo(sn_root)
            second = run_extract_reid_embeddings(
                crop_manifest=manifest,
                checkpoint=ckpt,
                checkpoint_sha256=_sha256_bytes(b"unit-test-checkpoint"),
                sn_reid_root=sn_root,
                expected_sn_reid_commit=commit,
                output_dir=out,
                batch_size=1,
                overwrite=True,
                model_builder=lambda **_: DeterministicModel(),
                weight_loader=lambda m, p: None,
            )
            self.assertEqual(second["crop_count"], 2)
            self.assertTrue((out / NPZ_NAME).is_file())
            parent = out.parent
            self.assertEqual(list(parent.glob("_tmp_reid_embeddings_*")), [])
            self.assertEqual(list(parent.glob("_backup_reid_embeddings_*")), [])
            self.assertGreater(second["elapsed_sec"], 0.0)
            self.assertEqual(first["status"], "ok")

    def test_temp_cleanup_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest, _ = _make_crop_bundle(tmp_path, n=1)
            ckpt = tmp_path / "ckpt.pth.tar"
            payload = b"x"
            ckpt.write_bytes(payload)
            sn_root = tmp_path / "sn-reid"
            commit = _init_sn_reid_repo(sn_root)
            out = tmp_path / "emb_out"

            def builder(*, model_name: str = MODEL_NAME):
                return ZeroNormModel()

            with self.assertRaises(EmbeddingError):
                run_extract_reid_embeddings(
                    crop_manifest=manifest,
                    checkpoint=ckpt,
                    checkpoint_sha256=_sha256_bytes(payload),
                    sn_reid_root=sn_root,
                    expected_sn_reid_commit=commit,
                    output_dir=out,
                    batch_size=1,
                    model_builder=builder,
                    weight_loader=lambda m, p: None,
                )
            self.assertFalse(out.exists())
            self.assertEqual(list(tmp_path.glob("_tmp_reid_embeddings_*")), [])


class SecurityFlagTests(unittest.TestCase):
    def test_build_model_flags_and_no_feature_extractor(self) -> None:
        fake_model = mock.MagicMock()
        fake_models = mock.MagicMock()
        fake_models.build_model.return_value = fake_model
        fake_utils = mock.MagicMock()
        fake_torchreid = mock.MagicMock()
        # Ensure FeatureExtractor attribute exists so we can assert it is unused.
        fake_utils.FeatureExtractor = mock.Mock()

        with mock.patch.dict(
            sys.modules,
            {
                "torchreid": fake_torchreid,
                "torchreid.models": fake_models,
                "torchreid.utils": fake_utils,
            },
        ):
            model = build_osnet_cpu_model()
            fake_models.build_model.assert_called_once_with(
                name="osnet_x1_0",
                num_classes=1,
                pretrained=False,
                use_gpu=False,
            )
            fake_model.eval.assert_called()
            fake_model.cpu.assert_called()
            self.assertIs(model, fake_model)
            fake_utils.FeatureExtractor.assert_not_called()
            fake_utils.load_pretrained_weights.assert_not_called()

            load_osnet_checkpoint_weights(fake_model, "/tmp/fake.pth.tar")
            fake_utils.load_pretrained_weights.assert_called_once()
            self.assertEqual(
                fake_utils.load_pretrained_weights.call_args[0][0], fake_model
            )
            fake_utils.FeatureExtractor.assert_not_called()

    def test_help_does_not_import_torchreid(self) -> None:
        import importlib.util

        script = _PROJECT_ROOT / "scripts" / "extract_reid_embeddings.py"
        spec = importlib.util.spec_from_file_location("extract_reid_embeddings_help", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Importing the CLI module should not require torchreid.
        with mock.patch.dict(sys.modules):
            # Remove torchreid if present so accidental import would fail.
            sys.modules.pop("torchreid", None)
            sys.modules.pop("torchreid.models", None)
            sys.modules.pop("torchreid.utils", None)
            spec.loader.exec_module(module)
            parser = module.build_parser()
            help_text = parser.format_help()
            self.assertIn("--crop-manifest", help_text)
            self.assertIn("--checkpoint-sha256", help_text)
            self.assertNotIn("torchreid", sys.modules)


if __name__ == "__main__":
    unittest.main()
