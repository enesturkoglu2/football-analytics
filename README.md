# Football Analytics

Görüntü işleme tabanlı futbolcu tespit, takip ve performans analizi
projesi.

## Proje amacı

Futbol videosunda:

1. Oyuncu detection ve tracking
2. Appearance tabanlı ReID (OSNet embedding)
3. Segment-level identity evidence (purity, kit, quality)
4. Jersey OCR (forma numarası kanıtı)
5. Daha sonra: target gallery memory ve spatial continuity

Jersey OCR, appearance ReID'in yerine geçmez; yalnız ek identity
evidence üretir. Forma numarası görünmediğinde appearance ReID
çalışmaya devam eder.

## Güncel pipeline özeti

```text
video
→ YOLO player detection
→ ByteTrack
→ crop quality / provenance
→ OSNet appearance embedding
→ segment construction
→ jersey visibility / ROI
→ DBNet text detection
→ SAR text recognition
→ segment-level evidence
→ future target gallery / fusion
```

## Güncel durum

Tamamlananlar:

- [x] ByteTrack tracking baseline
- [x] OSNet track-level ReID baseline (Stage 4B)
- [x] Crop quality / contamination ölçümü (Stage 5A)
- [x] Segmentation, purity audit ve segmented ReID regression (Stage 5B)
- [x] Jersey visibility / contamination / ROI ölçümü (Stage 5C)
- [x] 78-item manuel jersey pilot ve pilot freeze (Stage 5C-A)
- [x] DBNet/SAR capability audit (Stage 5C-B1/B2)
- [x] İzole MMOCR CPU environment kurulumu (Stage 5C-B4)
- [x] Kontrollü DBNet/SAR asset acquisition (Stage 5C-B5)

Henüz yapılmayanlar:

- [ ] DBNet/SAR model init
- [ ] Gerçek crop inference (jersey OCR smoke)
- [ ] Confidence threshold seçimi
- [ ] Segment-level OCR aggregation (Stage 5C-D)
- [ ] Target enrollment / gallery memory (Stage 5D)
- [ ] Evidence fusion evaluation (Stage 5E)
- [ ] Spatial continuity / pitch position (Stage 6)

DBNet/SAR checkpoint'leri yerel olarak edinildi fakat henüz
çalıştırılmadı; doğruluğu kanıtlanmış model olarak kabul edilmez.

## Geliştirme ortamları

| Environment | Amaç | Temel sürümler |
|---|---|---|
| `football-cv` | Ana video/detection/tracking/ReID geliştirme ve test suite | Python 3.10, torch 2.13.0+cpu, ultralytics, OpenCV |
| `sn-reid-cpu` | OSNet embedding inference (izole) | Python 3.10, torch 2.13.0+cpu |
| `sn-jersey-mmocr-cpu` | Jersey OCR smoke (izole, MMOCR stack) | Python 3.9, torch 1.13.1+cpu, mmocr 1.0.1, mmcv 2.0.1 |

Tümü CPU-only'dir; CUDA yoktur ve kurulmayacaktır.

Aktivasyon:

    conda activate football-cv
    cd ~/projects/football-analytics

## Ana dokümantasyon

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — ayrıntılı operasyonel durum,
  environment/repo/checkpoint bilgileri
- [docs/setup/stage5-identity-signals-plan.md](docs/setup/stage5-identity-signals-plan.md)
  — Stage 5 identity-signals planı ve kapı durumları
- [docs/setup/stage5c-jersey-pilot-results.md](docs/setup/stage5c-jersey-pilot-results.md)
  — 78-item manuel jersey pilot sonuçları
- [docs/setup/reid-stage4b-completion.md](docs/setup/reid-stage4b-completion.md)
  — Stage 4B ReID baseline kapanış raporu

## Güvenlik ve lisans notu

- DBNet/SAR asset'leri yalnız local research smoke için tutulur.
- Checkpoint redistribution/commercial durumu doğrulanmış değildir.
- Checkpoint'ler Git'e commit edilmez.
- SoccerNet jersey dataset ilk smoke için indirilmedi ve gerekli değil.
- Checkpoint path/SHA detayları için PROJECT_CONTEXT.md'ye bakınız.
