# Football Analytics — Proje Durumu ve Cursor Aktarım Raporu

## 1. Projenin amacı

Futbol videolarından oyuncu tespiti, oyuncu takibi, re-identification,
takım ve rol sınıflandırması, saha koordinatları, fiziksel metrikler,
top takibi, olay tespiti ve oyuncu raporları üretmek.

Nihai hedeflerden bazıları:

- Isı haritası
- Koşu mesafesi
- Sprint sayısı ve sprint hızı
- Pas başarı oranı
- Dripling
- İkili mücadele
- Hava topu
- Top çalma ve top kaybı
- Geçiş pasları
- Ceza sahasında topla buluşma
- Oyuncu aktiflik oranı

## 2. Geliştirme sistemi

- İşletim sistemi: Windows 11
- Geliştirme ortamı: WSL2
- Linux: Ubuntu 22.04.5 LTS
- WSL kernel: 6.6.87.2
- CPU: AMD Ryzen 5 5600X
- CPU thread: 12
- Bilgisayar RAM: 32 GB
- WSL tarafından görülen RAM: yaklaşık 15.6 GB
- GPU: Radeon RX 590 8 GB
- PyTorch GPU erişimi: yok
- Ana çalışma yöntemi: CPU

Not: WSL yaklaşık 1 TB disk gösterse de gerçek fiziksel boş alan
Windows C diskindeki yaklaşık 67 GB ile sınırlıdır.

## 3. Python ortamları

Üç izole conda environment kullanılmaktadır:

### 3.1 football-cv (ana geliştirme)

Python yolu:

    /home/enesturkoglu2/miniconda3/envs/football-cv/bin/python

- Python 3.10.20
- torch 2.13.0+cpu
- Ana video/detection/tracking/ReID geliştirme ortamı
- Mevcut test suite (443 test) bu ortamda çalışır

Ortam izolasyonu:

- PYTHONPATH temizlenir.
- PYTHONNOUSERSITE=1 ayarlanır.
- ROS Humble paketleri bu ortama karışmaz.
- ~/.local Python paketleri bu ortama karışmaz.

Aktivasyon:

    conda activate football-cv

### 3.2 sn-reid-cpu (OSNet/ReID)

- Python 3.10.20, torch 2.13.0+cpu, torchvision 0.28.0+cpu
- Yalnız OSNet embedding inference için kullanılır
- General ReID checkpoint (`osnet_x1_0` / Market1501) bu ortamla
  yüklenir; SoccerNet-trained değildir

### 3.3 sn-jersey-mmocr-cpu (jersey OCR smoke; Stage 5C-B4)

Path:

    /home/enesturkoglu2/miniconda3/envs/sn-jersey-mmocr-cpu

- Python 3.9.25
- torch 1.13.1+cpu
- torchvision 0.14.1+cpu
- numpy 1.26.4
- opencv-python 4.10.0.84
- mmcv 2.0.1 (prebuilt CPU wheel)
- mmengine 0.10.7
- mmdet 3.1.0
- mmocr 1.0.1
- CUDA false
- TrackLab / Hydra / EasyOCR / sn-gamestate package kurulu değildir
- Environment boyutu yaklaşık 1.6 GB

## 4. Kurulu temel paketler

- torch 2.13.0+cpu
- torchvision 0.28.0+cpu
- ultralytics 8.4.102
- opencv-python 5.0.0.93
- numpy 2.2.6
- lap 0.5.13
- pandas 2.3.3
- pyarrow 25.0.0
- matplotlib 3.10.9
- scipy 1.15.3
- scikit-learn 1.7.2
- pyyaml 6.0.3
- tqdm 4.69.0
- rich
- loguru
- FFmpeg 4.4.2

Son bağımlılık kontrolü:

    python -m pip check

Sonuç:

    No broken requirements found.

Ultralytics kontrolü başarılıdır:

- YOLO Python API çalışıyor.
- YOLO CLI çalışıyor.
- CPU inference kullanılacak.
- CUDA yoktur ve kurulmayacaktır.

## 5. Ana proje

Yerel proje yolu:

    /home/enesturkoglu2/projects/football-analytics

GitHub:

    git@github.com:enesturkoglu2/football-analytics.git

Ana dal:

    main

İlk commit:

    94daa9e Create project foundation and record environment

Güncel durum (REBUILD-R4D freeze / documentation commit):

- pre-commit HEAD: `b386f07c96782bcb595fbd7dd2fdfd696e491003`
- documentation/application commit: this REBUILD-R4D commit
  (hash after push; reported in the final gate report)
- `main == origin/main` after push
- tracked scope: nine recovery source/config/test files + two new
  docs + three updated docs (exact 14 files)
- sistem CPU-only'dir
- generated rebuild outputs remain Git-ignored / untracked

## 6. Proje klasörleri

- src/football_analytics
- scripts
- configs
- tests
- docs/setup
- data/test_clips
- outputs
- logs

Kurulum kayıtları docs/setup altında bulunmaktadır.

## 7. Tamamlanan aşama

Aşama 0 tamamlandı:

- WSL2 kurulumu
- Ubuntu 22.04 kurulumu
- Miniconda kurulumu
- football-cv ortamı
- Ortam izolasyonu
- PyTorch CPU kurulumu
- OpenCV ve veri paketleri
- Ultralytics kurulumu
- Git deposu
- GitHub SSH bağlantısı
- İlk commit ve push

Aşama 1 tamamlandı:

- Video yolu doğrulama
- FFprobe metadata (`outputs/ingest/ffprobe.json`)
- OpenCV okunabilirlik ve ilk kare kontrolü
- FPS, çözünürlük, kare sayısı, süre ve SHA-256
- `outputs/ingest/video_manifest.json`
- `--overwrite` koruması ve atomik JSON yazımı
- `unittest` ile ingest testleri
- Gerçek klip kabul testi: `data/test_clips/sample.mp4`

Aşama 2 tamamlandı:

- Temel insan/oyuncu adayı tespiti (COCO person, class 0)
- Model: `models/yolo11n.pt`
- Parametreler: `device=cpu`, `classes=[0]`, `conf=0.25`, `iou=0.70`, `imgsz=640`
- Tracking yok; `predict(..., save=False, verbose=False)`
- Benchmark: `outputs/detection/benchmark_100/` (100 kare)
- Tam video çıktıları: `outputs/detection/full/`
  - `annotated.mp4`
  - `detections.jsonl`
  - `detection_summary.json`
- Tam video metrikleri (`data/test_clips/sample.mp4`, 1023 kare):
  - frames_processed: 1023
  - total_detections: 14224
  - frames_with_detections: 1023
  - avg_detections_per_frame: yaklaşık 13.90
  - elapsed_sec: yaklaşık 66.31
  - avg_fps: yaklaşık 15.43
  - skipped_invalid: 0
- Annotated video: 1336x744, 30 FPS, 1023 kare
- `unittest` detection testleri dahil 36 test OK

Aşama 3 tamamlandı (temel ByteTrack tracking hattı):

- Person tracking + geçici `track_id` (kalıcı oyuncu kimliği değil)
- Tracker config: `configs/tracking/bytetrack_stage3.yaml`
- ByteTrack bağımlılığı: `lap==0.5.13`
- Detection parametreleri: `device=cpu`, `classes=[0]`, `conf=0.25`,
  `iou` / `detection_iou=0.70`, `imgsz=640`
- Tracker parametreleri (yaml varsayılanları): `tracker_type=bytetrack`,
  `track_high_thresh=0.25`, `track_low_thresh=0.1`, `new_track_thresh=0.25`,
  `track_buffer=30`, `match_thresh=0.8`, `fuse_score=True`
- `model.track(..., persist=True, save=False, verbose=False)`
- Benchmark: `outputs/tracking/benchmark_100/` (100 kare)
- Tam video çıktıları: `outputs/tracking/full/`
  - `tracked.mp4`
  - `tracks.jsonl`
  - `tracking_summary.json`
- Tam video metrikleri (`data/test_clips/sample.mp4`, 1023 kare):
  - frames_processed: 1023
  - total_box_observations: 13309
  - box_observations_with_track_id: 13309
  - box_observations_without_track_id: 0
  - unique_track_ids: 276
  - track_observation_count: min=1, max=371, mean≈48.22, median=9
  - track_span_frames: min=1, max=382, mean≈53.63, median=14
  - yalnızca 1 kare görülen ID: 72
  - ≤5 kare: 124; ≤10 kare: 148
  - skipped_invalid: 0
  - elapsed_sec: ≈61.82; avg_fps: ≈16.55
- Ground-truth yok: ID-switch / MOTA / HOTA ölçülmedi ve uydurulmadı
- 276 unique ID ve yüksek kısa ömürlü ID oranı tracking fragmentation
  işaretidir; bu ID'ler henüz güvenilir kalıcı oyuncu kimlikleri değildir
- Koşu sırasında bir kez `NMS time limit exceeded` uyarısı görüldü;
  işlem yine exit 0 / status=ok ile tamamlandı
- Ortam kaydı: `docs/setup/football-cv-after-tracking.txt` /
  `football-cv-after-tracking.yml`
- `unittest` 50 test OK (ingest + detection + tracking)

Aşama 4A tamamlandı (SoccerNet ReID altyapı hazırlığı; ürün ReID kodu yok):

- Dış repolar (ana `football-analytics` dışında, `~/projects/soccernet`):
  - SoccerNet SDK:
    `/home/enesturkoglu2/projects/soccernet/SoccerNet`
    pin: `74461027ac2095ce2f8d4ee991eccb5dd5f42459`
  - sn-reid (Torchreid fork):
    `/home/enesturkoglu2/projects/soccernet/sn-reid`
    pin: `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`
- İzole ortam: `sn-reid-cpu` (~1.7G)
  - Python 3.10.20, torch 2.13.0+cpu, torchvision 0.28.0+cpu
  - NumPy 2.2.6, OpenCV headless 5.0.0, CUDA False
  - `football-cv` bu aşamada değiştirilmedi
- Genel ReID checkpoint (SoccerNet-trained değil):
  - Mimari: `osnet_x1_0`
  - Eğitim: Market1501 same-domain person ReID (MODEL_ZOO)
  - Yol: `/home/enesturkoglu2/projects/soccernet/checkpoints/general-reid/osnet_x1_0_market1501_softmax_256x128.pth.tar`
  - Boyut: 10 399 605 bytes
  - SHA-256: `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`
  - Load: `pretrained=False` + `load_pretrained_weights`; 565 eşleşen;
    classifier 751→1 için 2 discard (beklenen); params 2 170 021
- Gerçek crop → embedding smoke (`outputs/reid/smoke/`, Git dışı):
  - track_id 463, frame 409, crop 58×114
  - embedding `(512,)`, float32, L2 ≈ 1.0; repeat max abs 0.0
  - FeatureExtractor kullanılmadı
- Büyük SoccerNet ReID dataseti indirilmedi
- SoccerNet SDK yalnız import bağımlılığı; `sample.mp4` için dataset aracı
  zorunlu değil
- Smoke kimlik doğruluğunu / track birleştirmeyi kanıtlamaz

Kurulum kayıtları: `docs/setup/sn-reid-*.md` / `sn-reid-cpu-after-install.*`

Aşama 4B tamamlandı (`completed_baseline`; track-level ReID / manual linking):

- Son ürün commit: `a4b379c` — Implement manual ReID linking pipeline
- `unittest` 165 test OK (ingest + detection + tracking + ReID)
- Model / checkpoint (4A ile aynı; SoccerNet-trained değil):
  - Mimari: `osnet_x1_0`
  - Checkpoint:
    `osnet_x1_0_market1501_softmax_256x128.pth.tar`
  - SHA-256:
    `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`
  - Eğitim: Market1501 general person ReID
  - `pretrained=false`; FeatureExtractor kullanılmadı; otomatik indirme yok
  - sn-reid pin: `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`
- Ortam sınırı: `football-cv` (crop/aggregate/candidates/linking/tests);
  `sn-reid-cpu` yalnız embedding inference
- Full `sample.mp4` sonuçları (`outputs/reid/full_stage4b/`, Git dışı):
  - 13 309 observation; 276 raw track
  - 454 crop; 135 crop/embedding track; 141 no-crop
  - crop embeddings `[454, 512]`; track embeddings `[135, 512]`;
    aggregation `l2_mean`; float32; L2-normalized
  - 9 045 candidate pair; exact-frame conflict 1 525;
    eligible_unthresholded 7 520
  - `similarity_threshold=null`; `automatic_linking_enabled=false`;
    cosine yalnız ranking/audit
  - 42 manual decision; 4 accepted edge
  - Accepted components:
    - `[4, 682]` → `global_candidate_id` 4
    - `[231, 635]` → `global_candidate_id` 231
    - `[593, 689]` → `global_candidate_id` 593
    - `[588, 806]` → `global_candidate_id` 588
  - 4 linked component; 8 linked raw track; 268 singleton
    (127 embedded unlinked + 141 no-embedding)
  - 276 raw → 272 global candidate
  - Final map:
    `outputs/reid/full_stage4b/linking/global_id_map.jsonl`
- Ham ByteTrack `track_id` değerleri yeniden yazılmaz
- ReID accuracy / MOTA / HOTA / IDF1 / ReID mAP yüzdesi hesaplanmadı ve
  uydurulmadı; global ID'ler candidate kimliktir, kanıtlanmış `player_id`
  değildir
- Kapanış raporu: `docs/setup/reid-stage4b-completion.md`
- Politika / şema:
  `docs/setup/reid-stage4b-linking-policy.md`,
  `docs/setup/reid-stage4b-schema-decisions.md`,
  `configs/reid/crop_selection_stage4b.yaml`,
  `configs/reid/linking_policy_stage4b.yaml`

Aşama 5A tamamlandı (crop quality / contamination ölçüm baseline):

- Ölçüm + görsel doğrulama; threshold seçilmedi, otomatik exclusion yok
- Status: `visually_validated_measurement_baseline`

Aşama 5B tamamlandı (kit ölçümü, purity audit, segmentation,
segmented ReID regression):

- Torso/kit descriptor ölçümü ve görsel doğrulama (takım ataması yok)
- Track purity audit + manuel non-destructive segment view
  (raw track'ler immutable; 13 split candidate, derived segment view)
- Segmented OSNet regression: 13 retired mixed parent, 28 recomputed
  manual segment, 122 reused embedding, 150 embedded segment entity
- Status: `completed_segmented_reid_regression_baseline`

Aşama 5C visibility/pilot tamamlandı (Stage 5C-A):

- Jersey visibility / contamination / ROI ölçümü (OCR'sız)
- Review panelleri ve 78-item manuel pilot (7 batch)
- Pilot freeze: `outputs/reid/full_stage4b/jersey_pilot_results_stage5c`
- Status: `completed_manual_review_pilot_baseline`
- Rapor: `docs/setup/stage5c-jersey-pilot-results.md`

Aşama 5C-B tamamlandı (jersey recognizer hazırlığı; model henüz
yüklenmedi):

- **B1/B2 capability audit:** sn-jersey ve sn-gamestate kontrollü clone
  + salt-okuma kod auditi; MMOCR (DBNet + SAR) primary candidate,
  EasyOCR fallback olarak seçildi
- **B3 environment/asset planı:** resmî package/checkpoint/license
  doğrulaması; PROFILE A minimal isolated MMOCR CPU planı onaylandı
- **B4 environment setup:** `sn-jersey-mmocr-cpu` kuruldu (bkz. bölüm
  3.3); import smoke geçti; model init yapılmadı
- **B5 asset acquisition:** DBNet/SAR config + checkpoint kontrollü
  indirildi ve doğrulandı (bkz. bölüm 10.2); checkpoint deserialize
  edilmedi
- Status: `completed_environment_and_assets_not_loaded`

### Stage 5C-C1 offline baseline smoke (completed)

- Status: `completed_offline_smoke_low_signal_baseline`
- Pipeline: successful (46/46, `inference_error=0`,
  network `pass_loopback_only`)
- Selection: POS/A/B/C/D/E = 20/10/2/7/2/5
- Positive exact match: 0/20 (smoke-set only; not a general accuracy
  benchmark)
- Detector no-region: 45; recognizer no-digit: 1; negative number
  emission: 0/26
- DBNet init ≈ 4736 ms; SAR init ≈ 1135 ms; peak RSS ≈ 1.5 GB
- Stage 5A number-search ROI reuse (median ROI ≈ 38×65 px)
- Provenance audit: `REPORT_TEXT_ONLY_TYPO`;
  `review_track_514_frame_496_rank_3` → manual jersey **30**
- Smoke root:
  `outputs/reid/full_stage4b/jersey_mmocr_smoke_stage5c`
- Baseline freeze:
  `outputs/reid/full_stage4b/jersey_mmocr_smoke_baseline_freeze_stage5c_c1`
- Application files:
  `src/football_analytics/reid/jersey_mmocr.py`,
  `scripts/run_reid_jersey_mmocr_smoke.py`,
  `configs/reid/jersey_mmocr_smoke_stage5c.yaml`,
  `tests/test_reid_jersey_mmocr.py`
- Results doc:
  `docs/setup/stage5c-jersey-mmocr-baseline-results.md`
- Technical finding: DBNet detector is the observed bottleneck on the
  current low-resolution number-search ROI baseline

### Stage 5C-C2 controlled ablation (completed)

- Status: `completed_no_exact_signal_in_tested_variants`
- Pipeline: successful (184/184, `inference_error=0`,
  `pass_loopback_only`)
- Exact matrix: `direct_sar_roi_1x`, `direct_sar_roi_2x_cubic`,
  `direct_sar_roi_4x_cubic`, `dbnet_sar_roi_4x_cubic`
- Exact match: 0/20 on all four new variants (smoke-set only; not a
  general accuracy benchmark)
- Direct SAR 1×/2×/4×: wrong low-confidence digit emissions only
  (3/2/2); negative emission 0/26
- DBNet+SAR 4×: region items 1→6, total regions 52; still exact 0/20
- Evidence labels: `UPSCALE_IMPROVES_DETECTION`,
  `NO_EXACT_SIGNAL_IN_TESTED_VARIANTS`
- Model-family status: general scene-text DBNet/SAR
  `closed_after_controlled_negative_result`
- Ablation freeze:
  `outputs/reid/full_stage4b/jersey_mmocr_ablation_freeze_stage5c_c2`
- Application files:
  `src/football_analytics/reid/jersey_mmocr.py` (ablation additions),
  `scripts/run_reid_jersey_mmocr_ablation.py`,
  `configs/reid/jersey_mmocr_ablation_stage5c_c2.yaml`,
  `tests/test_reid_jersey_mmocr_ablation.py`
- Results doc:
  `docs/setup/stage5c-jersey-mmocr-ablation-results.md`

### Stage 5C-C2a / C2b region review

- C2a package:
  `outputs/reid/full_stage4b/jersey_mmocr_detector_region_review_stage5c_c2a`
  — `review_package_generated_unreviewed` (52 regions)
- C2b: `skipped_by_project_decision_not_required_for_c3a`
  (working CSV blank; manual review not performed; localization
  failure not manually confirmed)

### Stage 5C-C3A PARSeq capability audit (completed; no install)

- External repo:
  `/home/enesturkoglu2/projects/external/jersey-number-pipeline`
  HEAD `007d54e5530a66616ed5081ca35e0028b36aadb5` (clean)
- SoccerNet fine-tuned PARSeq checkpoint metadata:
  Drive ID `1uRln22tlhneVt3P6MePmVxBWSLMsL3bm`;
  expected
  `parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt`;
  ~364M; `publicly_accessible_metadata_resolved`;
  official checksum unavailable; **not downloaded in C3A**
- CPU: `cpu_supported_with_small_adapter`
- Dataset: `dataset_not_required_for_initial_local_smoke`
- Stop/go: `GO_STAGE5C_C3B_ENV_PLAN`
- Audit doc: `docs/setup/stage5c-parseq-capability-audit.md`

### Stage 5C-C3B / C3C environment + checkpoint (completed)

- Environment: `sn-jersey-parseq-cpu` (isolated; CPU torch 1.13.1)
- Checkpoint local SHA-256:
  `14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879`
- Size: `381608677` bytes; official checksum still unavailable
- Generic `parseq-bb5792a6.pt` not used as runtime weight

### Stage 5C-C3D offline PARSeq smoke (completed)

- Frozen 46 ROI smoke: exact **5/20**, wrong **15/20**,
  no-prediction **0/20**; negative emission **26/26**
- Runtime contract validated; network `pass_loopback_only`
- Threshold **not** selected; not a deployment accuracy claim
- Freeze:
  `outputs/reid/full_stage4b/jersey_parseq_smoke_freeze_stage5c_c3d`
- Doc: `docs/setup/stage5c-parseq-smoke-results.md`

### Stage 5C-C3E false-positive / confidence audit (completed)

- Read-only analysis of C3D artifacts; no model reload
- Confidence descriptive ranking present
  (exact vs negative AUROC ≈0.938)
- Discovery-set perfect safe point observed; **not** independent
  validation and **not** a selected threshold
- `independent_positive_holdout_available=false`
  (all 20 readable positives used in C3D discovery set)
- Freeze:
  `outputs/reid/full_stage4b/jersey_parseq_false_positive_audit_freeze_stage5c_c3e`
- Doc: `docs/setup/stage5c-parseq-false-positive-audit.md`

## 8. Aktif kapı ve sıradaki adımlar

Güncel durum:

- **Rebuild r2 Stage 4B→Stage 5C recovery:** completed
  (historical structural counts exact; **not** historical freeze
  restore / byte-identity claim)
- **Canonical split generation:** `r2_capacity_balanced`
- **Canonical split root:**
  `outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced`
- Selected source quotas: reused/recomputed **110/18**; vector
  **5/7/4/2**; maximum feasible recomputed confirmed
- Previous R4 unbalanced split deprecated for downstream (immutable)
- Old 78-item pilot **not** reused as the r2 annotation set
- Old C3E threshold **not** reused
- Discovery primary annotation **not started**
- Holdout / reserve **unopened / unreviewed**
- Threshold / labels / predictions unseen
- **Sıradaki kapı:**
  `REBUILD-R5_STAGE5C_DISCOVERY_PRIMARY_ANNOTATION_FREEZE`
- DBNet/SAR family closed; jersey OCR **not** abandoned
- PARSeq remains the primary recognizer candidate
- Legibility classifier is a future helper-gate candidate
  (not installed/downloaded here)
- Stage 5D / 5E / 6 scopes unchanged

Docs:

- `docs/setup/rebuild-r2-stage4b-stage5-recovery.md`
- `docs/setup/stage5c-clean-discovery-holdout-design.md`

Planlanan sıra:

1. REBUILD-R5 discovery primary annotation freeze
2. Later confidence-gate / legibility validation only after
   discovery/holdout protocol advances
3. Stage 5C-D segment-level OCR aggregation
4. Stage 5D target gallery/enrollment
5. Stage 5E evidence fusion
6. Stage 6 spatial continuity / pitch position

## 9. Cursor için zorunlu kurallar

- Kullanıcıdan açık onay almadan paket kurma.
- Kullanıcıdan açık onay almadan repo klonlama.
- SoccerNet repolarının tamamını aynı anda kurma.
- Ana football-cv ortamına ağır dış repo bağımlılığı kurma.
- CUDA veya ROCm kurmaya çalışma.
- Mevcut torch ve torchvision sürümlerini değiştirme.
- Her kurulumdan önce dry-run ve sürüm kontrolü yap.
- Her önemli işlemden sonra python -m pip check çalıştır.
- Video ve dataset dosyalarını Git'e ekleme.
- Model ağırlıklarını Git'e ekleme.
- Her aşamayı bağımsız ve yeniden çalıştırılabilir tasarla.
- Hata oluştuğunda sonraki aşamaya geçme.
- Güvenilir olmayan analiz sonucunu uydurma; null ve confidence kullan.
- SoccerNet repolarını ana reponun içine kopyalama.
- Dış repolar ileride ~/projects/soccernet altında tutulacak.
- Her dış repo ayrı commit ve mümkünse ayrı Python ortamıyla kullanılacak.

## 10. SoccerNet durumu

### 10.1 Dış repolar (ana repo dışında; hepsi clean)

| Repo | Path | HEAD |
|---|---|---|
| SoccerNet SDK | `/home/enesturkoglu2/projects/soccernet/SoccerNet` | `74461027ac2095ce2f8d4ee991eccb5dd5f42459` |
| sn-reid | `/home/enesturkoglu2/projects/soccernet/sn-reid` | `621e2b0f2d2a7a3e207b8dd747542b6608bf72db` |
| sn-jersey | `/home/enesturkoglu2/projects/soccernet/sn-jersey` | `2f43b48c59eefe0bb5d948888db07f55f51208ad` |
| sn-gamestate | `/home/enesturkoglu2/projects/soccernet/sn-gamestate` | `1c958345067218297d221e45e1a6405f975f83e0` |

Jersey OCR yaklaşımı ile ilgili audit sonuçları (Stage 5C-B1/B2):

- `sn-jersey` reposu yalnız challenge/dataset README'sidir; çalışan
  recognizer kodu içermez.
- Jersey OCR yaklaşımı `sn-gamestate` auditinden çıkarılmıştır
  (MMOCR DBNet detector + SAR recognizer).
- Runtime'da sn-gamestate import etmeyen **clean MMOCR adapter**
  tercih edilmiştir; resmî MMOCR public API doğrudan kullanılacaktır.
- GPL-3.0 lisanslı sn-gamestate kodu football-analytics içine
  kopyalanmayacaktır.

Henüz kurulmayan / ertelenen adaylar (aynı anda kurulmayacak):

- sn-trackeval, sn-tracking, sn-calibration, sn-spotting,
  sn-teamspotting

Büyük SoccerNet datasetleri (jersey dataseti dahil) ve
SoccerNet-trained ReID checkpoint indirilmedi. Checkpoint klasörü:
`/home/enesturkoglu2/projects/soccernet/checkpoints/` (Git dışı).

### 10.2 DBNet/SAR asset'leri (Stage 5C-B5; Git dışı)

Asset root:

    /home/enesturkoglu2/projects/soccernet/checkpoints/jersey-mmocr

Top-level yapı: `dbnet/`, `sar/`, `configs/`, `manifests/`.

MMOCR source:

- Repo: `https://github.com/open-mmlab/mmocr.git`
- Tag: `v1.0.1`
- Commit: `1dcd6fa6958de22bcb997319833f0ac19c180ec7`

DBNet detector checkpoint:

- Path: `dbnet/dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth`
- Boyut: 59 068 395 byte
- SHA-256 (indirme sonrası bizim hesabımız; resmî tam checksum
  yayımlanmamıştır):
  `7c0e94f2f52e014fa423f489e059640e10f765f3c34f38b80454d7b850174cfb`

SAR recognizer checkpoint:

- Path: `sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real_20220915_171910-04eb4e75.pth`
- Boyut: 231 195 256 byte
- SHA-256 (indirme sonrası bizim hesabımız):
  `04eb4e75467fc951e9b189a273216dd2428d30f7d913368155ab5c69bbeb843c`

Toplam checkpoint boyutu: 290 263 651 byte.

Config closure (AST ile çıkarıldı; config execute edilmedi):

- 2 root config (DBNet + SAR)
- 21 dosya, 16 810 byte
- unresolved reference = 0
- SAR dictionary (`dicts/english_digits_symbols.txt`) dahildir

Manifestler:

- `manifests/asset_manifest.json` — SHA-256:
  `5b238ea7e83ed23a274329b2e811b3d26d319def8ba3b1a4434e46adfc4f62c2`
- `manifests/config_closure.json` — SHA-256:
  `fd29e4d262e80bce242274fe83ee19cc4e4ee0e6c021e60d9a67856b8f63d743`

Safety durumu:

- assets acquired and loaded for Stage 5C-C1 offline smoke only
- checkpoint loaded = true (local-path-only; no download)
- model initialized = true (CPU; DBNet + SAR)
- inference performed = true (46 crops; blind)
- baseline freeze status = `completed_offline_smoke_low_signal_baseline`
- dataset downloaded = false
- model fine-tuned = false

Lisans sınırı: framework Apache-2.0; checkpoint
redistribution/commercial durumu doğrulanmamıştır. Asset'ler yalnız
local research smoke içindir ve Git'e commit edilmez.

### 10.3 Pilot referans durumu (Stage 5C-A freeze)

- Canonical crop: 474
- Pilot item: 78 (78/78 reviewed)
- Valid crop: 65
- Number visible = yes: 26
- Readable = yes (jersey observation): 20
- Non-pilot unreviewed: 396

Sınırlar:

- Pilot bir accuracy benchmark'ı **değildir**.
- Single-reviewer (Furkan); independent double review yoktur.
- Crop-level observation'lar identity ground truth değildir.

Önemli purity/mixed-target bulguları:

- `raw_231` split 9/17 (iki segment farklı forma numarası gösterir)
- `raw_514` purity warning; frame 496 readable jersey observation = **30**
  (Stage 5C-C1a provenance audit: `REPORT_TEXT_ONLY_TYPO`)
- `raw_16` / `raw_13` invalid off-pitch crop kaynağı
- `raw_738` mixed target

## 11. Mevcut öncelik

Rebuild r2 Stage 4B→5C recovery ve clean capacity-balanced
discovery/holdout design tamamlandı. Canonical generation
`r2_capacity_balanced`. Eski 78-item pilot ve C3E threshold yeniden
kullanılmıyor. Discovery annotation henüz başlamadı; holdout
unopened. Sıradaki kapı
**REBUILD-R5_STAGE5C_DISCOVERY_PRIMARY_ANNOTATION_FREEZE**:

    Video okuma
    → İnsan/oyuncu tespiti (tamamlandı)
    → ByteTrack geçici takip (tamamlandı; fragmented)
    → ReID altyapı hazırlığı (4A tamamlandı)
    → Track-level ReID / linking baseline (4B tamamlandı)
    → Crop quality / kit / purity / segmentation (5A-5B tamamlandı)
    → Jersey visibility + 78-item manuel pilot (5C-A historical;
      not r2 annotation set)
    → MMOCR environment + DBNet/SAR asset (5C-B tamamlandı)
    → Offline CPU jersey OCR baseline smoke (5C-C1 tamamlandı)
    → Controlled ablation (5C-C2 tamamlandı; no exact signal;
      DBNet/SAR family closed; jersey OCR not abandoned)
    → PARSeq capability audit (5C-C3A tamamlandı; no install)
    → Isolated PARSeq CPU environment (5C-C3B tamamlandı)
    → Controlled PARSeq checkpoint acquisition (5C-C3C tamamlandı)
    → Offline PARSeq smoke (5C-C3D tamamlandı; exact 5/20;
      negative emission 26/26; no threshold)
    → PARSeq false-positive / confidence audit (5C-C3E tamamlandı;
      descriptive ranking; C3E threshold not reused)
    → Rebuild r2 Stage 4B→5C recovery + clean 474 universe
      + capacity-balanced split (tamamlandı; r2_capacity_balanced)
    → Discovery primary annotation freeze (REBUILD-R5 — sıradaki)
    → Segment aggregation / gallery / fusion (5C-D, 5D, 5E)
    → Spatial continuity (Stage 6)

## 12. Jersey OCR güvenlik kısıtları (Stage 5C-C sonrası da geçerli)

- Checkpoint/model asset'leri Git'e commit edilmez.
- MMOCR model alias'ı ile otomatik download yapılmaz; yalnız local
  config ve local weight path kullanılacaktır.
- Checkpoint lisans/redistribution belirsizliği korunur; asset'ler
  yalnız local research smoke içindir.
- SoccerNet jersey dataset indirilmedi; ilk smoke için gerekli
  değildir.
- Hiçbir jersey OCR sonucu doğrudan identity ground truth sayılmaz;
  jersey OCR evidence'tır, identity değildir.
- Stage 5A/5B artifact'leri yeniden üretilmeden reuse edilir.
- Yeni videoda detection/tracking doğal olarak tekrar çalışacaktır.
- Identity/gallery/team/global-ID işlemleri jersey smoke kapılarında
  değiştirilmez.
