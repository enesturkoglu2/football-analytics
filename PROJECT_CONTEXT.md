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

## 3. Python ortamı

Conda ortamı:

    football-cv

Python yolu:

    /home/enesturkoglu2/miniconda3/envs/football-cv/bin/python

Python sürümü:

    3.10.20

Ortam izolasyonu:

- PYTHONPATH temizlenir.
- PYTHONNOUSERSITE=1 ayarlanır.
- ROS Humble paketleri bu ortama karışmaz.
- ~/.local Python paketleri bu ortama karışmaz.

Aktivasyon:

    conda activate football-cv

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

## 8. Sıradaki aşama

Aşama 4B baseline kapandı. Önerilen sonraki iyileştirmeler (ayrı onay):

- Jersey number recognition ve/veya team/kit classification
- Crop contamination detection
- Temporal/spatial motion consistency
- Football-domain ReID fine-tuning
- Labelled golden clips + resmi değerlendirme metrikleri

Takım sınıflandırması ve kalıcı oyuncu kimliği ürün hattı hâlâ ayrı
aşamalardır. Ayrı onay olmadan yeni ürün kodu yazılmayacaktır.

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

Aşama 4A ile klonlanan / pinlenen dış repolar (ana repo dışında):

1. SoccerNet Python SDK — `74461027ac2095ce2f8d4ee991eccb5dd5f42459`
2. sn-reid — `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`

Henüz kurulmayan / ertelenen adaylar (aynı anda kurulmayacak):

- sn-trackeval, sn-tracking, sn-calibration, sn-jersey, sn-gamestate,
  sn-spotting, sn-teamspotting

Büyük SoccerNet datasetleri ve SoccerNet-trained ReID checkpoint
indirilmedi. Checkpoint klasörü:
`/home/enesturkoglu2/projects/soccernet/checkpoints/` (Git dışı).

## 11. Mevcut öncelik

Aşama 4B track-level ReID baseline (`completed_baseline`) tamamlandı.
Önerilen sonraki adım: jersey/team kimlik sinyalleri veya bir sonraki ana
proje aşaması (ayrı onay):

    Video okuma
    → İnsan/oyuncu tespiti (tamamlandı)
    → ByteTrack geçici takip (tamamlandı; fragmented)
    → ReID altyapı hazırlığı (4A tamamlandı)
    → Track-level ReID / linking baseline (4B tamamlandı)
    → Jersey/team sinyalleri veya sonraki ana aşama (ayrı onay)
    → İşaretlenmiş video
