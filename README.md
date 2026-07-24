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
→ SoccerNet-finetuned PARSeq recognizer (primary candidate)
→ future legibility / holdout / aggregation gates
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
- [x] Offline CPU DBNet+SAR baseline smoke (Stage 5C-C1)
- [x] Controlled recognizer/preprocessing ablation (Stage 5C-C2)
- [x] PARSeq jersey capability audit (Stage 5C-C3A; no install)
- [x] Isolated PARSeq CPU environment (Stage 5C-C3B)
- [x] Controlled SoccerNet PARSeq checkpoint acquisition (Stage 5C-C3C)
- [x] Offline PARSeq recognizer-only smoke (Stage 5C-C3D)
- [x] PARSeq false-positive / confidence audit (Stage 5C-C3E)
- [x] Rebuild r2 Stage 4B→5C recovery (historical structural counts
      exact; historical freeze restore değildir)
- [x] Clean label-blind 474-item universe + capacity-balanced
      discovery/holdout design (`r2_capacity_balanced`)
- [x] Discovery primary annotation freeze (REBUILD-R5; reserve closed)
- [x] Discovery-primary PARSeq inference + zero-error candidate gate
      (REBUILD-R6; support=1; not a deployment threshold)

Henüz yapılmayanlar:

- [ ] Holdout primary annotation freeze (REBUILD-R7)
- [ ] Holdout evaluation against frozen candidate gate
- [ ] Confidence threshold seçimi / validation (deployment)
- [ ] Segment-level OCR aggregation (Stage 5C-D)
- [ ] Target enrollment / gallery memory (Stage 5D)
- [ ] Evidence fusion evaluation (Stage 5E)
- [ ] Spatial continuity / pitch position (Stage 6)

DBNet/SAR ailesi C2 sonrası kapalıdır; jersey OCR iptal edilmedi.
SoccerNet-finetuned PARSeq primary candidate olmaya devam eder.
C3D/C3E historical pilot sonuçları referans olarak durur; **eski
78-item pilot yeniden annotation seti değildir** ve **eski C3E
threshold yeniden kullanılmaz**. Recovery sonrası canonical split
generation: **`r2_capacity_balanced`**. Discovery-primary annotation
tamamlandı (10 readable / 27 non-readable; reserve açılmadı).
Discovery PARSeq: exact 5/5 wrong; negative digit emission **27/27**;
zero-error candidate cut exact **`0.99992299168434329`** (`>=`,
support=1, item `discovery_primary_028`). Bu cut **deployment /
calibration claim değildir**; holdout sonucuna göre değiştirilemez.
Holdout hâlâ unopened/unreviewed. Sıradaki kapı:
**REBUILD-R7_STAGE5C_HOLDOUT_PRIMARY_ANNOTATION_FREEZE**.
Legibility classifier ileride yardımcı gate adayıdır (henüz
kurulmadı/indirilmedi). Stage 5D/5E/6 kapsamları değişmedi.
Checkpoint/asset'ler Git'e commit edilmez.

## Geliştirme ortamları

| Environment | Amaç | Temel sürümler |
|---|---|---|
| `football-cv` | Ana video/detection/tracking/ReID geliştirme ve test suite | Python 3.10, torch 2.13.0+cpu, ultralytics, OpenCV |
| `sn-reid-cpu` | OSNet embedding inference (izole) | Python 3.10, torch 2.13.0+cpu |
| `sn-jersey-mmocr-cpu` | Legacy DBNet/SAR smoke (izole, MMOCR stack; family closed) | Python 3.9, torch 1.13.1+cpu, mmocr 1.0.1, mmcv 2.0.1 |
| `sn-jersey-parseq-cpu` | PARSeq jersey recognizer smoke (izole) | Python 3.9, torch 1.13.1+cpu, pytorch-lightning 1.9.5, timm 0.9.5 |

Tümü CPU-only'dir; CUDA yoktur ve kurulmayacaktır.

Aktivasyon:

    conda activate football-cv
    cd ~/projects/football-analytics

## Ana dokümantasyon

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — ayrıntılı operasyonel durum,
  environment/repo/checkpoint bilgileri
- [docs/setup/stage5-identity-signals-plan.md](docs/setup/stage5-identity-signals-plan.md)
  — Stage 5 identity-signals planı ve kapı durumları
- [docs/setup/rebuild-r2-stage4b-stage5-recovery.md](docs/setup/rebuild-r2-stage4b-stage5-recovery.md)
  — Stage 4B→5C rebuild r2 recovery / provenance
- [docs/setup/stage5c-clean-discovery-holdout-design.md](docs/setup/stage5c-clean-discovery-holdout-design.md)
  — Clean label-blind capacity-balanced discovery/holdout design
- [docs/setup/stage5c-discovery-parseq-candidate-gate-r2.md](docs/setup/stage5c-discovery-parseq-candidate-gate-r2.md)
  — Discovery-primary PARSeq candidate gate (r2; not deployment)
- [docs/setup/stage5c-jersey-pilot-results.md](docs/setup/stage5c-jersey-pilot-results.md)
  — 78-item manuel jersey pilot sonuçları (historical; not r2 annotation set)
- [docs/setup/stage5c-jersey-mmocr-baseline-results.md](docs/setup/stage5c-jersey-mmocr-baseline-results.md)
  — Stage 5C-C1 offline DBNet+SAR baseline smoke sonuçları
- [docs/setup/stage5c-jersey-mmocr-ablation-results.md](docs/setup/stage5c-jersey-mmocr-ablation-results.md)
  — Stage 5C-C2 controlled ablation sonuçları ve model-family kapanışı
- [docs/setup/stage5c-parseq-capability-audit.md](docs/setup/stage5c-parseq-capability-audit.md)
  — Stage 5C-C3A SoccerNet fine-tuned PARSeq capability audit
- [docs/setup/stage5c-parseq-smoke-results.md](docs/setup/stage5c-parseq-smoke-results.md)
  — Stage 5C-C3D offline PARSeq smoke sonuçları
- [docs/setup/stage5c-parseq-false-positive-audit.md](docs/setup/stage5c-parseq-false-positive-audit.md)
  — Stage 5C-C3E PARSeq confidence / false-positive audit
- [docs/setup/reid-stage4b-completion.md](docs/setup/reid-stage4b-completion.md)
  — Stage 4B ReID baseline kapanış raporu

## Güvenlik ve lisans notu

- DBNet/SAR asset'leri yalnız local research smoke için tutulur.
- Checkpoint redistribution/commercial durumu doğrulanmış değildir.
- Checkpoint'ler Git'e commit edilmez.
- SoccerNet jersey dataset ilk smoke için indirilmedi ve gerekli değil.
- Checkpoint path/SHA detayları için PROJECT_CONTEXT.md'ye bakınız.
