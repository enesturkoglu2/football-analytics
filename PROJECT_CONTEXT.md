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

## 8. Sıradaki aşama

Aşama 1: Video ingest ve metadata.

Bu aşamada:

1. Kullanıcı tarafından verilen video doğrulanacak.
2. FFprobe ile metadata alınacak.
3. OpenCV ile videonun okunabilirliği kontrol edilecek.
4. FPS, çözünürlük, kare sayısı ve süre raporlanacak.
5. SHA-256 checksum üretilecek.
6. JSON manifest yazılacak.

Henüz model inference yapılmayacak.

Beklenen çıktı:

    outputs/ingest/video_manifest.json
    outputs/ingest/ffprobe.json

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

Henüz hiçbir SoccerNet reposu bu bilgisayarda kurulmadı veya klonlanmadı.

Planlanan kullanım sırası:

1. SoccerNet Python SDK
2. sn-trackeval
3. sn-tracking
4. sn-calibration
5. sn-reid
6. sn-jersey
7. sn-gamestate
8. sn-spotting
9. sn-teamspotting

Bu sıra proje ihtiyaçlarına göre değişebilir. Hepsi aynı anda kurulmayacaktır.

## 11. Mevcut öncelik

Öncelik SoccerNet kurulumu değil, kendi kısa videomuz üzerinde şu hattın
çalıştığını doğrulamaktır:

    Video okuma
    → İnsan/oyuncu tespiti
    → Oyuncu takibi
    → Geçici oyuncu ID'leri
    → İşaretlenmiş video
