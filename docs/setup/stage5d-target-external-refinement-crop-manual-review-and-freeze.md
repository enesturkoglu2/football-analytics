# Stage 5D-F3F — External Refinement Crop Manual Review and Freeze

## Purpose

Freeze exact human crop decisions from the F3E review package:

- EXT_161 target crops: all 4 → `target_crop_yes`
- Hard-negative crops: 23 yes / 4 wrong-team no / 5 invalid / 3 ambiguous

No embeddings, gallery-v2, hard-negative membership, OCR/similarity,
threshold, identity assignment, or sample access.

## Exact HN sequence mapping

Sequence numbers map to `target_001_ext_hard_negative_candidate_NNN`.

| Decision | Sequences |
|---|---|
| hard_negative_crop_yes | 002,005,006,007,008,011,012,014,015,018–021,024–027,029–031,033–035 |
| hard_negative_crop_no (wrong-team) | 004,013,016,017 |
| invalid | 001,010,023,028,032 |
| multi_person_ambiguous | 003,009,022 |

## Status

`COMPLETED_STAGE5D_F3F_TARGET_001_EXTERNAL_REFINEMENT_CROP_DECISIONS_FROZEN`

Exact next gate:
`STAGE5D-F3G_TARGET_001_APPROVED_CROP_OSNET_EMBEDDING_AND_GALLERY_V2_BUILD`

## Run

```bash
conda run -n football-cv python scripts/run_reid_external_refinement_crop_manual_freeze.py \
  --config configs/reid/external_refinement_crop_manual_freeze_stage5d_target_001.yaml
```
