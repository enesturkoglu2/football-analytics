# Football Analytics

Görüntü işleme tabanlı futbolcu tespit, takip ve performans analizi projesi.

## İlk hedef

Kısa bir futbol videosunda:

1. Videoyu doğrulamak
2. Oyuncuları tespit etmek
3. Oyunculara geçici takip kimliği vermek
4. İşaretlenmiş video ve standart veri çıktıları üretmek

## Geliştirme ortamı

- Windows 11
- WSL2
- Ubuntu 22.04
- Conda ortamı: football-cv
- Python 3.10
- PyTorch CPU
- OpenCV
- Ultralytics

## Ortamı etkinleştirme

Terminalde aşağıdaki komutlar çalıştırılır:

    conda activate football-cv
    cd ~/projects/football-analytics

## Proje durumu

- [x] WSL2 ve Ubuntu kurulumu
- [x] İzole Conda ortamı
- [x] PyTorch CPU doğrulaması
- [x] OpenCV ve veri paketleri
- [x] Ultralytics doğrulaması
- [x] Video ingest ve metadata
- [x] Oyuncu detection
- [x] Oyuncu tracking
- [x] Aşama 4A ReID altyapı hazırlığı (izole sn-reid-cpu + genel ReID smoke)
- [ ] Aşama 4B track-level ReID / linking
- [ ] Takım sınıflandırması
- [ ] Kalıcı oyuncu re-identification ürün hattı
- [ ] Kamera kalibrasyonu
- [ ] Oyuncu metrikleri
