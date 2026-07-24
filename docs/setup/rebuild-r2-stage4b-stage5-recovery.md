# Rebuild r2 — Stage 4B → Stage 5C recovery

- **Gate:** REBUILD-R4D freeze / documentation / commit
- **Canonical split generation:** `r2_capacity_balanced`
- **Project HEAD at design freeze (pre-commit):**
  `b386f07c96782bcb595fbd7dd2fdfd696e491003`
- **This document does not claim historical output restore or
  byte-identity with any pre-loss Stage 4B/5C trees.**

## 1. Output kaybının kapsamı

WSL/Windows recovery sonrası, generated Stage 4B / Stage 5 ürün
root'ları yeniden üretildi. Bu süreç **historical freeze restore
değildir**: amaç, aynı source contract ve documented replay
kuralları altında r2 provenance kökleri oluşturmaktır.

Kaybolan / yeniden üretilen ürün kökleri (örnek):

- Stage 4B ReID / linking root
- documented-link overlay
- Stage 5A/5B/5B3 replay
- Stage 5C visibility / clean-review universe
- Stage 5C clean discovery/holdout split

Kaynak video, YOLO/OSNet checkpoint'leri ve Git history bu rebuild
içinde yeniden indirilmedi / rewrite edilmedi.

## 2. Source video contract

- Path: `data/test_clips/sample.mp4`
- Development / reference clip only (ürün sabiti değildir)
- Rebuild r2 tüm Stage 4B→5C zinciri bu klip üzerinde
  yeniden koşturulmuştur

## 3. YOLO / OSNet asset contract

- Detection: `models/yolo11n.pt` (CPU; person class)
- Appearance ReID: `osnet_x1_0` / Market1501 general checkpoint
  (SoccerNet-trained değildir)
- Embedding inference: izole `sn-reid-cpu` ortamı
- Checkpoint'ler Git'e commit edilmez; rebuild sırasında
  network/download yapılmadı

## 4. Tracking historical exact match

Rebuild r2 tracking/observation zinciri, historical Stage 3/4B
structural counts ile exact eşleşecek şekilde doğrulandı
(fragmented ByteTrack `raw_track_id` korunur; kalıcı `player_id`
değildir).

## 5. Documented four-edge replay

Manual linking accepted components (documented four edges)
yeniden uygulandı:

- `[4, 682]`
- `[231, 635]`
- `[593, 689]`
- `[588, 806]`

Automatic linking kapalıdır; similarity threshold null kalır.

## 6. Global candidate count

- **272** global candidate (276 raw track → 4 linked components /
  documented merge sonrası historical structural count)

## 7. Stage 5A / 5B / 5B3 replay

- Stage 5A: crop quality / contamination measurement replay
- Stage 5B: kit descriptor measurement replay
- Stage 5B3: purity audit + manual non-destructive segmentation +
  segmented ReID regression replay
- Threshold seçilmedi; otomatik exclusion / global-ID rewrite yok

## 8. Segment structural counts

Historical exact structural counts (r2 replay):

- **291** segments
- **13301** assigned observations
- **8** unassigned observations

## 9. Segmented ReID historical exact counts

Segmented ReID regression, historical exact count sözleşmesiyle
doğrulandı (embedding/aggregation contract korunur; accuracy claim
üretilmez).

## 10. Visibility / canonical review items

- **474** Stage 5C visibility / clean-review universe item
- Label-blind; ROI-valid; annotation_status=`unreviewed`
- Manual jersey labels / OCR predictions universe seçiminde
  kullanılmaz

## 11. Historical freeze restore değildir

Rebuild r2:

- historical output byte-identity iddiası taşımaz
- eski `outputs/reid/full_stage4b` path'ini geri getirmez
- C3D/C3E membership / threshold reuse yapmaz
- 78-item manuel pilot'u yeniden annotation seti olarak kullanmaz

## 12. r2 provenance kökleri (aktif)

| Role | Path |
|---|---|
| Stage 4B base | `outputs/reid/full_stage4b_rebuild_r2` |
| Documented-link overlay | `outputs/reid/full_stage4b_rebuild_r2_documented_link_overlay` |
| Stage 5 replay | `outputs/reid/full_stage4b_rebuild_r2_stage5_replay` |
| Visibility / clean universe | `outputs/reid/full_stage4b_rebuild_r2_stage5c_visibility_universe` |
| Deprecated initial split (R4) | `outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split` |
| **Canonical capacity-balanced split** | `outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced` |

## 13. NTFS snapshot listesi (recovery)

Windows NTFS:

`/mnt/c/Users/enest/Documents/football_analytics_recovery/rebuild_snapshots`

Önemli arşivler (extract edilmeden saklanır):

- `REBUILD_R2_STAGE4B_STAGE5_b386f07.tar.gz`
- `REBUILD_R3_STAGE5C_UNIVERSE_b386f07.tar.gz`
- `REBUILD_R4_CLEAN_SPLIT_PRE_REBALANCE_b386f07.tar.gz`
- `REBUILD_R4B_FAILED_CAPACITY_CODE_b386f07.tar.gz`
- `REBUILD_R4C_CAPACITY_BALANCED_SPLIT_b386f07.tar.gz`

## 14. Active downstream readiness

Downstream annotation / OCR kapıları yalnız canonical capacity-balanced
split root üzerinden açılır:

`outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced`

- `canonical_split_generation=r2_capacity_balanced`
- Discovery primary annotation **henüz başlamadı**
- Holdout / reserve paketleri **unopened / unreviewed**
- Threshold seçilmedi; predictions görülmedi
- Exact next gate:
  `REBUILD-R5_STAGE5C_DISCOVERY_PRIMARY_ANNOTATION_FREEZE`

## 15. Code classification (recovery sonrası dokuz dosya)

### Canonical aktif

- Clean review universe:
  - `scripts/build_reid_jersey_clean_review_universe.py`
  - `configs/reid/jersey_clean_review_universe_stage5c_rebuild_r2.yaml`
  - `tests/test_reid_jersey_clean_review_universe.py`
- Clean split (capacity-balanced):
  - `scripts/build_reid_jersey_clean_split.py`
  - `configs/reid/jersey_clean_split_stage5c_rebuild_r2.yaml`
  - `tests/test_reid_jersey_clean_split.py`

### Preserved but superseded

Eski C3F-A holdout tasarım üçlüsü (silinmedi):

- `scripts/build_reid_jersey_parseq_holdout.py`
- `configs/reid/jersey_parseq_holdout_stage5c_c3f_a.yaml`
- `tests/test_reid_jersey_parseq_holdout.py`

Sözleşme:

- `superseded_by` =
  `outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced`
- `must_not_be_used_for_new_r2_annotation=true`
- `retained_for_history_and_tests=true`
