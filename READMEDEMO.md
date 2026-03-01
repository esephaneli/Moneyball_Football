# Moneyball Football 
### From a Movie to a Model — YOLOv11 + ByteTrack ile Gerçek Zamanlı Oyuncu Takibi

> **"Sahadadaki her hareketi sayıya dökebilsek?"**  
> Bu proje, Moneyball felsefesini futbola uyguluyor — maç görüntüsünden doğrudan oyuncu performans verisi çıkarmak için Computer Vision kullanıyor.

---

## Demo

![Demo Çıktısı](demo_preview.gif)

> Inter vs Genoa — Serie A 2024–25 | YOLOv11n + ByteTrack | Hareket izli gerçek zamanlı takip

---

## Ne Yapıyor?

- Maç videosunu kare kare işleyerek **oyuncuları** ve **topu** tespit ediyor
- ByteTrack ile her oyuncuya **benzersiz bir ID** atıyor ve frame'ler boyunca takip ediyor
- Son 30 kareyi kullanarak **gradient hareket izi** çiziyor
- Sol üst köşeye **canlı bilgi paneli** ekliyor (frame, oyuncu sayısı, top)
- Tüm anotasyonlarla birlikte **MP4 çıktısı** üretiyor

---

## Mimari

```
FootballAnalytics          (Orkestratör)
├── Config                 Tüm parametreler tek yerden
├── VideoProcessor         Video okuma / yazma
├── ObjectDetector         YOLO inference + sınıf filtreleme
├── PlayerTracker          ID bazlı durum yönetimi + trajectory geçmişi
├── Visualizer             Görselleştirme — saf rendering, iş mantığı yok
└── StatisticsModule       [Faz 2 — stub] Hız, mesafe, sprint tespiti
```

Her sınıfın tek bir sorumluluğu var. Herhangi bir bileşeni değiştirmek diğerlerini etkilemiyor.

---

## Faz 2 — Yakında

| Özellik | Durum |
|---|---|
| Anlık hız (km/h) | Planlandı |
| Toplam koşulan mesafe (m) | Planlandı |
| Sprint tespiti ve sayımı | Planlandı |
| Homography ile piksel → metre kalibrasyonu | Planlandı |
| Forma rengine göre takım sınıflandırması | Planlandı |
| Oyuncu bazlı ısı haritası | Planlandı |

`StatisticsModule` stub'ı kodun içinde — Faz 2'nin nereye entegre edileceğini gösteriyor.

---

## Kullanılan Teknolojiler

| Araç | Amaç |
|---|---|
| Python 3.12 | Ana dil |
| Ultralytics YOLOv11 | Nesne tespiti |
| ByteTrack | Çoklu nesne takibi |
| OpenCV | Video I/O ve görselleştirme |
| NumPy | Sayısal işlemler |
| uv | Paket yönetimi |

---

## Hızlı Başlangıç

**1. Bağımlılıkları yükle**
```bash
uv pip install ultralytics opencv-python numpy
```

**2. Videoyu ekle**
```
money_ball/
├── demo.py
├── input.mp4      <- maç videosu buraya
└── output.mp4     <- işlenmiş çıktı (otomatik oluşur)
```

**3. Çalıştır**
```bash
uv run demo.py
```

YOLOv11n modeli (~6MB) ilk çalıştırmada otomatik indirilir.

---

## Konfigürasyon

Tüm parametreler `Config` sınıfında — kod içinde arama yapmaya gerek yok:

```python
class Config:
    MODEL        = "yolo11n.pt"    # yolo11s.pt daha iyi doğruluk
    CONFIDENCE   = 0.35
    TRACKER      = "bytetrack.yaml"
    DEVICE       = ""              # "" = otomatik | "cuda" | "mps" | "cpu"
    TRAIL_LEN    = 30              # hareket izi uzunluğu (frame)
    TARGET_CLS   = {"person", "sports ball"}
    INPUT_VIDEO  = "input.mp4"
    OUTPUT_VIDEO = "output.mp4"
```

---

## Notlar

- **Yüksek track ID'leri** (#200+) broadcast görüntülerinde normaldir — kamera açısı değişimlerinde ByteTrack yeni ID atıyor. Faz 2'de Re-ID ile çözülecek.
- **Top tespiti** oyuncuların ayağında gizli kaldığında başarısız olabiliyor — tek model yaklaşımının bilinen bir kısıtlaması.
- Bu dosya bir **mimari demo**'dur. İstatistik hesaplama modülü dahil tam implementasyon paylaşılmamıştır.

---

## İlham

> *"Scouts'ların yerini almaya çalışmıyoruz. Onlara daha iyi bilgi vermeye çalışıyoruz."*  
> — Moneyball (2011)

Aynı fikir, bir kamera ve sinir ağıyla futbola uygulandı.

---

*Python & OpenCV ile geliştirildi | Devam eden bir spor analitiği projesinin parçası*
