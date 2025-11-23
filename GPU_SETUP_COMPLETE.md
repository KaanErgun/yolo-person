# ✅ GPU Setup Complete - RTX 3090

## 🎉 Proje Başarıyla NVIDIA RTX 3090 için Yapılandırıldı!

### 📊 Sistem Özeti

**GPU Bilgileri:**
- GPU: NVIDIA GeForce RTX 3090
- VRAM: 24 GB
- CUDA Sürümü: 12.6
- Driver Sürümü: 560.94

**PyTorch CUDA Durumu:**
- CUDA Kullanılabilir: ✅ Evet
- CUDA Sürümü: 12.8
- GPU Sayısı: 1

### ⚙️ Yapılandırma Ayarları (.env)

```bash
DEVICE=cuda                    # NVIDIA GPU kullanılıyor
BATCH_SIZE=48                  # RTX 3090 için optimize edilmiş
WORKERS=8                      # Çoklu işlem desteği aktif
EPOCHS=100                     # Tam eğitim için 100 epoch
AMP=true                       # Mixed precision training aktif
IMAGE_SIZE=640                 # Standart YOLO boyutu
MODEL_SIZE=s                   # YOLOv10-small (dengeli)
```

### 🚀 Kullanım

#### 1. Dataset İndirme
```bash
source venv/bin/activate
python download_coco.py       # ~19GB - zaman alacak
python prepare_dataset.py     # Person sınıfını filtrele
```

#### 2. Training Başlatma
```bash
source venv/bin/activate
python train.py
```

#### 3. GPU Monitoring
Başka bir terminal'de:
```bash
./monitor_gpu.sh              # Real-time GPU izleme
# veya
watch -n 1 nvidia-smi         # Klasik nvidia-smi monitoring
```

#### 4. Inference (Eğitim Sonrası)
```bash
# Webcam
python inference.py --source 0

# Görsel
python inference.py --source image.jpg

# Video
python inference.py --source video.mp4
```

### 📈 Beklenen Performans (RTX 3090)

| Metrik | Değer |
|--------|-------|
| Batch Size | 48 |
| Training Hızı | ~1.5-2.0 batch/s |
| Epoch Süresi | ~25-30 dakika |
| 100 Epoch Toplam | ~42-50 saat |
| VRAM Kullanımı | ~18-20 GB |

### 🔧 DNS Sorunu Çözüldü

WSL2'de DNS sorunu vardı, Google DNS eklenerek çözüldü:
```bash
# /etc/resolv.conf'a eklendi:
nameserver 8.8.8.8
nameserver 8.8.4.4
```

### 📝 Önemli Notlar

1. **Batch Size Ayarlama**: Eğer OOM (Out of Memory) hatası alırsanız:
   ```bash
   # .env dosyasında BATCH_SIZE'ı düşürün
   BATCH_SIZE=32  # veya 24, 16
   ```

2. **Training İzleme**: 
   - TensorBoard: `tensorboard --logdir runs/train`
   - Logs: `runs/train/yolov10_person/`
   - Checkpoints: `runs/train/yolov10_person/weights/`

3. **En İyi Model**: Training sonunda:
   - `runs/train/yolov10_person/weights/best.pt` - En yüksek mAP
   - `runs/train/yolov10_person/weights/last.pt` - Son epoch

4. **Devam Ettirme**: Training kesintiye uğrarsa:
   ```python
   # config.py'de RESUME=True yapın
   # veya train.py'de resume parametresi kullanın
   ```

### 🛠️ Troubleshooting

**Problem: CUDA out of memory**
```bash
# Çözüm 1: Batch size azalt
BATCH_SIZE=32

# Çözüm 2: Image size küçült
IMAGE_SIZE=512
```

**Problem: Training çok yavaş**
```bash
# Workers sayısını artır (dikkat: RAM kullanımı artar)
WORKERS=12

# Cache aktif et (RAM'de dataset önbellekleme)
CACHE=ram
```

**Problem: GPU kullanılmıyor**
```bash
# CUDA durumunu kontrol et
python -c "import torch; print(torch.cuda.is_available())"

# .env dosyasında DEVICE=cuda olduğundan emin ol
```

### 📚 Ek Komutlar

```bash
# GPU durumunu kontrol et
nvidia-smi

# Detaylı GPU bilgisi
nvidia-smi -q

# Config'i görüntüle
python config.py

# Setup testi
python test_setup.py

# Sistem temizliği (eğer baştan başlamak isterseniz)
rm -rf runs/train/*
```

### 🎯 Sonraki Adımlar

1. ✅ GPU setup tamamlandı
2. ⏳ Dataset indiriliyor (devam ettirilmeli)
3. ⏹️ Dataset hazırlama (prepare_dataset.py)
4. ⏹️ Training başlatma (train.py)
5. ⏹️ Model değerlendirme
6. ⏹️ Inference testleri

---

**🎊 Tebrikler! Projeniz RTX 3090 ile training için tamamen hazır!**

Training başlatmadan önce dataset indirmesinin tamamlanmasını bekleyin.
