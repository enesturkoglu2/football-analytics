# Stage 5C-C1 — Offline DBNet+SAR jersey OCR baseline smoke results

- **Date:** 2026-07-22
- **Stage status:** `completed_offline_smoke_low_signal_baseline`
- **Pipeline status:** successful
- **Model signal status:** `detector_low_signal_on_current_roi`
- **Provenance audit (5C-C1a):** `REPORT_TEXT_ONLY_TYPO`
- **Next gate:** Stage 5C-C2 controlled recognizer/preprocessing ablation

## 1. Amaç ve kapsam

Bu kapı, Stage 5C-B5'te edinilen local DBNet+SAR asset'leriyle
**offline CPU** jersey OCR smoke'unu çalıştırdı ve sonucu immutable
bir baseline freeze paketinde dondurdu.

Kapsam:

- 46 deterministik pilot crop
- Stage 5A number-search ROI reuse (ayrı ROI dosyası yok)
- Blind inference + ayrı evaluation
- Loopback-only network policy
- Identity/gallery/team/global-ID değişikliği yok

Kapsam dışı:

- preprocessing / upscaling / threshold tuning
- direct SAR (detector bypass) deneyi
- EasyOCR fallback
- dataset indirme
- model fine-tune
- segment aggregation
- C2 ablation (henüz başlatılmadı)

## 2. Environment ve local asset

Environment: `sn-jersey-mmocr-cpu`

| Paket | Sürüm |
|---|---|
| Python | 3.9.25 |
| torch | 1.13.1+cpu |
| torchvision | 0.14.1+cpu |
| numpy | 1.26.4 |
| opencv-python | 4.10.0.84 |
| mmcv | 2.0.1 |
| mmengine | 0.10.7 |
| mmdet | 3.1.0 |
| mmocr | 1.0.1 |
| CUDA | false |

Asset root (Git dışı):

`/home/enesturkoglu2/projects/soccernet/checkpoints/jersey-mmocr`

| Model | Checkpoint SHA-256 |
|---|---|
| DBNet `dbnet_resnet18_fpnc_1200e_icdar2015` | `7c0e94f2…174cfb` |
| SAR `sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real` | `04eb4e75…be843c` |

Asset manifest SHA-256:
`5b238ea7e83ed23a274329b2e811b3d26d319def8ba3b1a4434e46adfc4f62c2`

Config closure SHA-256:
`fd29e4d262e80bce242274fe83ee19cc4e4ee0e6c021e60d9a67856b8f63d743`

## 3. Offline / network politikası

`unshare -n` bu sistemde `Operation not permitted`; zorunlu şart
olarak kullanılmadı (`unshare_supported=false`).

İzolasyon:

- geçici `HOME` / `XDG_CACHE_HOME` / `TORCH_HOME` / `TMPDIR`
- geçersiz HTTP/HTTPS/ALL proxy
- `strace -f -yy -s 256 -e trace=network`

Sonuç: `policy_status=pass_loopback_only`

- loopback socket = 1, loopback bind = 1 (`::1`)
- external connect/send = 0
- DNS / wildcard bind = 0
- automatic download attempted = false

## 4. Deterministik 46-item selection (düzeltilmiş taxonomy)

Kaynak: Stage 5C-A pilot freeze
(`jersey_pilot_results_stage5c/pilot_reviewed_items.jsonl`).

| Sınıf | Adet | Kural özeti |
|---|---:|---|
| `POS_readable` | 20 | valid + visible=yes + readable=yes + nonblank jersey |
| `A_not_visible` | 10 | valid + visible=no; ≤1/segment; contamination öncelikli |
| `B_visible_unreadable` | 2 | valid + visible=yes + readable=no (strict) |
| `C_uncertain_signal` | 7 | valid + (visible=uncertain **veya** readable=uncertain) |
| `D_uncertain_crop` | 2 | crop_valid=uncertain (pilot_index ascending) |
| `E_invalid` | 5 | crop_valid=invalid (pilot_index ascending) |

Önceki taxonomy hatası: frozen veride `visible=yes/readable=no`
yalnız **2** item vardır; dört `readable=uncertain` item **C**
sınıfındadır.

Duplicate review_item_id / pilot_index / category overlap = 0.

## 5. Stage 5A ROI reuse

- ROI kaynağı: Stage 5A number-search ROI metadata
- Preprocessing variant: `number_search_roi_v1` / BGR in-memory crop
- Median ROI yaklaşık **38×65 px**
- Ayrı ROI image export yok; source crop overwrite yok

## 6. DBNet/SAR initialization

Local-path-only init (URL/alias reddi; CPU zorunlu):

| Aşama | Exact değer |
|---|---:|
| DBNet init | 4735.75 ms |
| SAR init | 1134.55 ms |
| Peak RSS | 1 539 352 KB (~1.5 GB) |

Checkpoint load warning (fail değil):

`unexpected key in source state_dict: data_preprocessor.mean, data_preprocessor.std`

(DBNet + SAR; config `data_preprocessor` mean/std runtime'da mevcut.)

Ayrıca mmengine registry scope warning ve LocalVisBackend `save_dir`
UserWarning kaydedildi. `inference_error_count=0`.

## 7. Prediction sonuçları

**Pipeline sonucu (başarılı):**

- offline initialization successful
- 46/46 inference completed
- `inference_error=0`
- source/provenance valid
- network policy passed

**Model sinyal sonucu (zayıf, dürüst):**

- DBNet 45/46 item'da text region üretmedi
- Tek region üreten item'da SAR adayları digit olarak kabul edilmedi
  (`raw_text='I'` → `non_digit_text`; letter→digit dönüşümü yok)
- Readable positives exact match = **0/20**
- Negatives number emission = **0/26**

Ana teknik çıkarım:

> DBNet detector is the observed bottleneck on the current
> low-resolution number-search ROI baseline.

Bu ifade “model tamamen işe yaramaz”, “SAR başarısızlığı kanıtlandı”
veya “jersey OCR mümkün değil” anlamına gelmez. SAR yalnız region
oluşan tek item'da çalıştırıldı; detector-bypass deneyi yapılmadı.

### Positive metrikler (20)

| Metrik | Değer |
|---|---:|
| exact_match_count | 0 |
| exact_match_rate | 0.0 |
| prediction_count | 0 |
| wrong_number_count | 0 |
| no_prediction_count | 20 |

### Negative / safety (26)

| Sınıf | Items | Number emission | Emission rate |
|---|---:|---:|---:|
| A_not_visible | 10 | 0 | 0.0 |
| B_visible_unreadable | 2 | 0 | 0.0 |
| C_uncertain_signal | 7 | 0 | 0.0 |
| D_uncertain_crop | 2 | 0 | 0.0 |
| E_invalid | 5 | 0 | 0.0 |

### Sayaçlar

| Sayaç | Değer |
|---|---:|
| detector_no_region_count | 45 |
| recognizer_no_digit_count | 1 |
| inference_error_count | 0 |
| false_positive_number_count | 0 |

## 8. Runtime

| Ölçüm | Exact ms |
|---|---:|
| Item total median | 656.22 |
| Item total mean | 718.33 |
| Item total p95 | 838.81 |
| Detector item median | 650.45 |

## 9. C1a provenance audit

Kritik item: `review_track_514_frame_496_rank_3`

| Alan | Değer |
|---|---|
| pilot_index | 36 |
| segment_id | `raw_514_s02` |
| raw_track_id | 514 |
| frame_index | 496 |
| manual_crop_valid | valid |
| manual_number_visible | yes |
| manual_number_readable | yes |
| manual_digit_count | 2 |
| manual_jersey_number | **30** |
| prediction | null |
| selection_class | POS_readable |
| source_crop_sha256 | `50db48c3…d2be17` |

Doğrulama:

- frozen CSV = 30
- reviewed JSONL = 30
- evaluation reference = 30
- item evaluation = 30
- mismatch = 0
- yalnız önceki kullanıcı rapor metninde `17` yazılmıştı
- artifact değişikliği gerekmedi

`raw_514` crop-level seri (identity ground truth değildir):

| Segment | Frame | Visible | Readable | Jersey |
|---|---:|---|---|---|
| raw_514_s01 | 446 | yes | yes | 3 |
| raw_514_s01 | 454 | yes | yes | 8 |
| raw_514_s02 | 463 | yes | yes | 30 |
| raw_514_s02 | 479 | yes | yes | 30 |
| raw_514_s02 | 487 | yes | no | (boş) |
| raw_514_s02 | 496 | yes | yes | 30 |

## 10. Interpretation limits

- Yalnız 46 deterministik pilot crop
- Median ROI ≈ 38×65 px
- Preprocessing / upscaling / full-person fallback yapılmadı
- Direct recognizer deneyi yapılmadı
- Threshold seçilmedi
- Segment aggregation yapılmadı
- Identity accuracy ölçülmedi
- Küçük smoke set genel accuracy benchmark **değildir**
- Readable reference crop-level insan observation'ıdır
- Dataset indirilmedi; model fine-tune edilmedi
- Sonuç identity ground truth değildir

## 11. Baseline freeze artifact'leri

Path:

`outputs/reid/full_stage4b/jersey_mmocr_smoke_baseline_freeze_stage5c_c1`

(Git-ignored; PNG/JPEG/video/crop/checkpoint yok.)

| Dosya | SHA-256 | Byte | Rows |
|---|---|---:|---:|
| smoke_inference_manifest.jsonl | `f6d8ffa9…c2abf39` | 30818 | 46 |
| smoke_evaluation_reference.jsonl | `fd168460…512022` | 17216 | 46 |
| smoke_predictions.jsonl | `8a413b63…df1087` | 81284 | 46 |
| smoke_item_evaluation.jsonl | `8c8f6949…ed4daa` | 26198 | 46 |
| smoke_results_summary.json | `1df90131…cd07e41` | 4347 | — |
| smoke_runtime_summary.json | `82365531…d0b8f66e` | 2714 | — |
| smoke_run_manifest.json | `ed68889b…4af4ee85` | 5813 | — |
| smoke_network_audit.txt | `24ae5185…feea7d844` | 375 | 15 |
| baseline_freeze_summary.json | `f6874f4e…c86cfc3` | 8113 | — |
| baseline_freeze_manifest.json | `852ea56d…b13471` | 5309 | — |

Sekiz smoke artifact, source smoke root ile **byte-identical**tir.

## 12. Uygulama dosyaları (repo)

- `src/football_analytics/reid/jersey_mmocr.py`
- `scripts/run_reid_jersey_mmocr_smoke.py`
- `configs/reid/jersey_mmocr_smoke_stage5c.yaml`
- `tests/test_reid_jersey_mmocr.py`

Sözleşme özeti: local-path-only init, URL/alias reddi, CPU
enforcement, checkpoint/config/crop SHA doğrulama, blind inference,
ayrı evaluation, digit-only `[0-9]{1,2}`, threshold yok,
letter→digit yok, loopback-only network audit.

## 13. Sonraki kapı gerekçesi

Baseline, pipeline'ın çalıştığını ve mevcut düşük çözünürlüklü
number-search ROI üzerinde **DBNet detector'ın gözlenen darboğaz**
olduğunu gösterdi. Sıradaki kapı:

**Stage 5C-C2 — controlled recognizer/preprocessing ablation**

Bu kapıda C2 henüz çalıştırılmamıştır.
