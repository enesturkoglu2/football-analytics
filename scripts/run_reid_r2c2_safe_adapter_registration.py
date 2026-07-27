#!/usr/bin/env python3
"""ReID-R2C2 integration smoke + audit artifact builder (no scoring)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.embedding import (  # noqa: E402
    MODEL_ID_SPORTSREID_SOCCERNET,
    embed_image_paths_with_model,
    load_reid_osnet_by_model_id,
)
from football_analytics.reid.model_registry import get_reid_model_spec  # noqa: E402

FINAL_OUT = PROJECT / "outputs/reid/target_001_reid_r2c2_safe_adapter_registration"
SMOKE_MANIFEST = (
    PROJECT
    / "outputs/reid/target_001_reid_r2c_checkpoint_security_review"
    / "smoke_crop_selection_manifest.jsonl"
)
EXPECTED_SMOKE_MANIFEST_SHA = (
    "6bd0d8b1ec6d16a4df9ad065065de2bba49440d005f65876b868dc87eb5d055b"
)
MARKET_CKPT = Path(
    "/home/enesturkoglu2/projects/soccernet/checkpoints/general-reid/"
    "osnet_x1_0_market1501_softmax_256x128.pth.tar"
)
FULL_SPORTS = Path(
    "/home/enesturkoglu2/projects/soccernet/checkpoints/"
    "_quarantine_sportsreid_osnet_20260727T205544_17463/osnet_checkpoint.pth.tar"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if FINAL_OUT.exists():
        print("BLOCKED_R2C2_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    unique = f"{int(time.time())}_{Path.cwd().name}"
    tmp = PROJECT / f"outputs/reid/_tmp_target_001_reid_r2c2_safe_adapter_registration_{unique}"
    tmp.mkdir(parents=True, exist_ok=False)

    head = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()
    spec = get_reid_model_spec(MODEL_ID_SPORTSREID_SOCCERNET)
    ckpt = Path(str(spec["checkpoint_path"]))
    assert ckpt.is_file()
    assert sha256_file(ckpt) == str(spec["sha256"]).lower()
    assert ckpt.stat().st_size == int(spec["size_bytes"])

    # Smoke manifest integrity
    assert SMOKE_MANIFEST.is_file()
    man_sha = sha256_file(SMOKE_MANIFEST)
    assert man_sha == EXPECTED_SMOKE_MANIFEST_SHA, (man_sha, EXPECTED_SMOKE_MANIFEST_SHA)
    rows = [
        json.loads(line)
        for line in SMOKE_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
    for row in rows:
        crop = Path(row["crop_path"])
        assert crop.is_file()
        assert sha256_file(crop) == str(row["crop_sha256"]).lower()

    # Execution path checks before load
    assert "football_analytics.reid.multiframe_r2b" not in sys.modules

    loaded = load_reid_osnet_by_model_id(MODEL_ID_SPORTSREID_SOCCERNET)
    assert loaded["weights_only"] is True
    assert loaded["allowlist_used"] is False
    assert loaded["checkpoint_sha256"] == str(spec["sha256"]).lower()
    assert Path(loaded["checkpoint_path"]).resolve() == ckpt.resolve()
    assert "football_analytics.reid.multiframe_r2b" not in sys.modules

    paths = [row["crop_path"] for row in rows]
    e1 = embed_image_paths_with_model(loaded["model"], paths, batch_size=2)
    e2 = embed_image_paths_with_model(loaded["model"], paths, batch_size=2)
    assert e1.shape == (6, 512)
    maxdiff = float(np.max(np.abs(e1 - e2)))
    norms = np.linalg.norm(e1, axis=1)
    nan = int(np.isnan(e1).sum())
    inf = int(np.isinf(e1).sum())
    zero = int(np.sum(norms < 1e-6))
    assert maxdiff == 0.0
    assert nan == 0 and inf == 0 and zero == 0
    assert bool(np.isfinite(e1).all())
    assert all(abs(float(n) - 1.0) < 1e-4 for n in norms)

    # Access audit: Market1501 and full SportsReID must not be read by this smoke.
    access = {
        "sanitized_checkpoint_accessed": True,
        "sanitized_checkpoint_path": str(ckpt),
        "sanitized_checkpoint_sha256": sha256_file(ckpt),
        "original_full_sportsreid_used_in_inference": False,
        "market1501_checkpoint_used_in_inference": False,
        "market1501_path": str(MARKET_CKPT),
        "full_sportsreid_path": str(FULL_SPORTS),
        "network": False,
        "package_change": False,
        "r2b_import": False,
        "r2b_runner": False,
    }

    coverage = loaded["architecture_coverage"]
    (tmp / "architecture_key_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    (tmp / "checkpoint_load_validation.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
                "weights_only": True,
                "allowlist_used": False,
                "checkpoint_path": loaded["checkpoint_path"],
                "checkpoint_sha256": loaded["checkpoint_sha256"],
                "success": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp / "adapter_execution_path.json").write_text(
        json.dumps(
            {
                "entrypoint": "scripts/run_reid_r2c2_safe_adapter_registration.py",
                "loader": "football_analytics.reid.safe_checkpoint.load_sportsreid_osnet_cpu_model",
                "model_selection": "load_reid_osnet_by_model_id",
                "preprocessing": "football_analytics.reid.embedding.load_and_preprocess_crop",
                "modules_imported_project_local": [
                    "football_analytics.reid.embedding",
                    "football_analytics.reid.model_registry",
                    "football_analytics.reid.safe_checkpoint",
                ],
                "multiframe_r2b_imported": False,
                "r2b_runner_executed": False,
                "wildcard_import": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp / "model_registry_registration.json").write_text(
        json.dumps({"registered_model": spec, "registry_path": "configs/reid/model_registry.yaml"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (tmp / "network_and_asset_access_audit.json").write_text(
        json.dumps(access, indent=2) + "\n", encoding="utf-8"
    )

    with (tmp / "integration_smoke_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "smoke_index": i,
                        "class": row["class"],
                        "stable_query_id": row["stable_query_id"],
                        "crop_path": row["crop_path"],
                        "crop_sha256": row["crop_sha256"],
                        "embedding_dim": 512,
                        "l2_norm": float(norms[i]),
                        "embedding_sha256": hashlib.sha256(
                            e1[i].astype(np.float32).tobytes()
                        ).hexdigest(),
                    }
                )
                + "\n"
            )
    (tmp / "integration_smoke_determinism.json").write_text(
        json.dumps(
            {
                "smoke_crop_count": 6,
                "embedding_dim": 512,
                "pass1_pass2_max_abs_diff": maxdiff,
                "nan_count": nan,
                "inf_count": inf,
                "zero_vector_count": zero,
                "l2_norms": [float(x) for x in norms],
                "selection_manifest_sha256": man_sha,
                "active_checkpoint_sha256": loaded["checkpoint_sha256"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = {
        "gate": "REID_R2C2_CONTROLLED_SAFE_ADAPTER_PATCH_AND_REGISTRATION",
        "head": head,
        "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
        "weights_only": True,
        "allowlist": False,
        "scoring_115": False,
        "gallery_modified": False,
    }
    (tmp / "effective_r2c2_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )

    status = "COMPLETED_R2C2_SAFE_ADAPTER_AND_MODEL_REGISTRATION"
    next_gate = "REID_R2D_SOCCERNET_OSNET_SINGLE_CROP_DOMAIN_ABLATION"
    manifest = {
        "schema_version": "target_001_reid_r2c2_manifest_v1",
        "final_status": status,
        "exact_next_gate": next_gate,
        "head_before_commit": head,
        "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
        "sanitized_checkpoint_sha256": loaded["checkpoint_sha256"],
        "sanitized_size_bytes": ckpt.stat().st_size,
        "weights_only": True,
        "allowlist_used": False,
        "original_full_checkpoint_used_in_inference": False,
        "smoke_max_abs_diff": maxdiff,
        "query_115_scoring": False,
        "gallery_gt_modified": False,
    }
    (tmp / "target_001_reid_r2c2_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    ignored = coverage.get("ignored_classifier_keys") if coverage else []
    report = f"""# Target 001 — ReID-R2C2 Safe Adapter Registration Report

## 1. Final status
- **Status:** `{status}`
- **Exact next gate:** `{next_gate}`
- Bu kapı performans ölçmedi (AP/AUROC/Recall yok).
- 115-query scoring yapılmadı.
- Gallery/GT değiştirilmedi.
- Automatic identity assignment false.
- Original full SportsReID checkpoint inference'da kullanılmadı.
- Production SportsReID loader: `weights_only=True`, allowlist=false.
- Market1501 loader davranışı değiştirilmedi.
- Commit/push bu script dışında yapılır.

## 2. Model
- model_id: `{MODEL_ID_SPORTSREID_SOCCERNET}`
- architecture: osnet_x1_0
- input: 256×128
- embedding_dim: 512
- checkpoint SHA-256: `{loaded['checkpoint_sha256']}`

## 3. Architecture coverage
- decision: `{coverage.get('decision') if coverage else None}`
- backbone_coverage: {coverage.get('backbone_coverage') if coverage else None}
- ignored classifier keys: {ignored}

## 4. Integration smoke
- 6 crops reused from security-review manifest SHA `{man_sha}`
- max-abs-diff: {maxdiff}
- NaN/Inf/zero: {nan}/{inf}/{zero}

## 5. Execution path
- R2B import/run: false
- Active loader: safe_checkpoint + embedding.load_reid_osnet_by_model_id
"""
    (tmp / "target_001_reid_r2c2_safe_adapter_registration_report.md").write_text(
        report, encoding="utf-8"
    )

    shutil.move(str(tmp), str(FINAL_OUT))
    print(json.dumps({"final_status": status, "next_gate": next_gate, "maxdiff": maxdiff}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
