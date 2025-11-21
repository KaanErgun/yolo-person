# 🚀 YOLOv10 Person Detection Training on Apple Silicon

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![YOLOv10](https://img.shields.io/badge/YOLOv10-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)

> A production-ready YOLOv10 training pipeline optimized for **Apple Silicon (M4)** using the COCO "person" subset. Features environment-based configuration, professional project structure, and comprehensive documentation.

## ✨ Key Highlights

- 🎯 **Specialized Training**: Focused person detection using 64,115 COCO images
- 🍎 **Apple Silicon Optimized**: Native MPS (Metal Performance Shaders) support for M1/M2/M3/M4 chips
- ⚙️ **Environment-Based Config**: Clean `.env` configuration system for easy deployment
- 📊 **COCO Pipeline**: Automated dataset download, filtering, and YOLO format conversion
- 🔄 **Production Ready**: Professional project structure with MIT license and contribution guidelines
- 📈 **Performance Metrics**: ~0.58 batch/s on M4 with 19.4GB GPU utilization

## 📊 Performance Benchmarks

| Device | Model | Batch Size | Speed | GPU Memory | Time/Epoch |
|--------|-------|------------|-------|------------|------------|
| M4 (24GB) | YOLOv10s | 24 | 0.58 batch/s | 19.4GB | ~77 min |
| M4 (24GB) | YOLOv10n | 32 | 0.72 batch/s | 15.2GB | ~62 min |

**Estimated Training Time**: ~5.3 days for 100 epochs (YOLOv10s on M4)

## 🎯 What's Inside

### Core Features

```
✅ COCO Dataset Integration    → Automated download and preprocessing
✅ Person Class Filtering       → 64,115 train + 2,693 validation images
✅ YOLO Format Conversion       → Normalized bounding box annotations
✅ Apple MPS Backend           → Native GPU acceleration for M-series chips
✅ Environment Configuration   → Flexible .env-based settings
✅ Training Pipeline           → Complete YOLOv10 training workflow
✅ Inference Scripts           → Image, video, and webcam support
```

### Project Architecture

```
yolo-person/
├── 📄 .env.example            # Configuration template
├── 📄 config.py               # Configuration loader with validation
├── 📄 dataset.yaml            # YOLO dataset specification
├── 📄 download_coco.py        # COCO 2017 downloader
├── 📄 prepare_dataset.py      # COCO→YOLO converter (person filter)
├── 📄 train.py                # Main training script
├── 📄 inference.py            # Inference runner
├── 📄 requirements.txt        # Python dependencies
├── 📄 LICENSE                 # MIT License
├── 📄 CONTRIBUTING.md         # Contribution guidelines
│
├── 📁 data/                   # Raw COCO dataset (gitignored)
│   └── coco/
│       ├── train2017/
│       ├── val2017/
│       └── annotations/
│
├── 📁 datasets/               # Processed YOLO format (gitignored)
│   └── coco_person/
│       ├── images/
│       └── labels/
│
├── 📁 runs/                   # Training outputs (gitignored)
│   ├── train/
│   └── detect/
│
└── 📁 samples/                # Demo outputs (kept in git)
    └── .gitkeep
```

## 🚀 Quick Start

### ⚡ Automated Setup (Recommended)

**New to this project? Run the automated setup script:**

```bash
git clone https://github.com/yourusername/yolo-person.git
cd yolo-person
./first-run.sh
```

The script will automatically:
- ✅ Check system requirements
- ✅ Create `.env` configuration with optimal settings
- ✅ Set up Python virtual environment
- ✅ Install all dependencies
- ✅ Run system tests

---

### 🔧 Manual Setup

If you prefer manual setup:

### Prerequisites

- **Python**: 3.9 or higher
- **RAM**: 16GB minimum, 24GB+ recommended
- **Storage**: 150GB for full COCO dataset + processed data
- **GPU**: Apple Silicon (M1/M2/M3/M4) or NVIDIA CUDA

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/yolo-person.git
cd yolo-person
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Configuration

The project uses `.env` for all configurable parameters. Key settings:

```bash
# Model Configuration
MODEL_SIZE=s                    # n, s, m, l, x (nano to xlarge)
TRAINING_EPOCHS=100
BATCH_SIZE=24                   # Adjust based on GPU memory
IMAGE_SIZE=640

# Device Settings
DEVICE=mps                      # mps, cuda, cpu, or device ID
WORKERS=0                       # Set to 0 for MPS compatibility
AMP_ENABLED=false               # Mixed precision (disable for MPS stability)

# Paths
DATASET_PATH=./datasets/coco_person
COCO_PATH=./data/coco
PROJECT_PATH=./runs/train
```

See `.env.example` for full configuration options with detailed comments.

## 📖 Usage Guide

### Step 1: Download COCO Dataset

Download COCO 2017 train/val splits (~19GB):

```bash
python download_coco.py
```

**Options:**
```bash
python download_coco.py --no-images      # Skip images (annotations only)
python download_coco.py --data-dir ./my_data
```

### Step 2: Prepare Person Dataset

Filter and convert COCO to YOLO format:

```bash
python prepare_dataset.py
```

**Output:**
- 64,115 training images with person annotations
- 2,693 validation images
- Normalized YOLO format: `<class> <x_center> <y_center> <width> <height>`

**Options:**
```bash
python prepare_dataset.py --coco-dir ./data/coco --output-dir ./datasets/custom
```

### Step 3: Train YOLOv10

Start training with your configured settings:

```bash
python train.py
```

**Training progress:**
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/100     19.4G      1.234      0.856      1.123        128        640
2/100     19.4G      1.156      0.782      1.067        128        640
...
```

**Outputs:**
```
runs/train/yolov10_person/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last epoch checkpoint
├── results.png          # Training curves
├── confusion_matrix.png
├── PR_curve.png
└── F1_curve.png
```

### Step 4: Run Inference

Use your trained model for predictions:

```bash
# Webcam
python inference.py --source 0

# Image
python inference.py --source path/to/image.jpg

# Video
python inference.py --source path/to/video.mp4

# Directory
python inference.py --source path/to/images/

# Custom model + visualize
python inference.py --model runs/train/yolov10_person/weights/best.pt --source test.jpg --show
```

## 🔧 Advanced Configuration

### Model Size Selection

| Model | Parameters | Speed | mAP | Use Case |
|-------|------------|-------|-----|----------|
| YOLOv10n | 2.3M | ⚡⚡⚡ | 🎯🎯 | Mobile, edge devices |
| YOLOv10s | 7.2M | ⚡⚡ | 🎯🎯🎯 | **Balanced (recommended)** |
| YOLOv10m | 15.4M | ⚡ | 🎯🎯🎯🎯 | High accuracy applications |
| YOLOv10l | 24.4M | 🐌 | 🎯🎯🎯🎯 | Research, benchmarking |
| YOLOv10x | 29.5M | 🐌🐌 | 🎯🎯🎯🎯🎯 | Maximum accuracy |

Edit `.env`:
```bash
MODEL_SIZE=m  # Change to your preferred size
```

### Apple Silicon Optimization

**Key settings for M-series chips:**

```bash
DEVICE=mps                 # Use Metal Performance Shaders
WORKERS=0                  # Disable multiprocessing (MPS limitation)
AMP_ENABLED=false          # Disable mixed precision (stability)
CACHE_ENABLED=true         # Cache validation set in memory
BATCH_SIZE=24              # Optimal for M4 24GB (adjust for your RAM)
```

**Memory considerations:**
- M1 8GB: `BATCH_SIZE=8`
- M2 16GB: `BATCH_SIZE=16`
- M3/M4 24GB: `BATCH_SIZE=24-32`
- M3 Max 36GB+: `BATCH_SIZE=48+`

### Training Hyperparameters

All tunable via `.env`:

```bash
# Optimization
OPTIMIZER=AdamW             # AdamW, Adam, SGD
LEARNING_RATE=0.001
MOMENTUM=0.937
WEIGHT_DECAY=0.0005

# Regularization
DROPOUT=0.0
LABEL_SMOOTHING=0.0

# Augmentation
HSV_H=0.015                 # Hue augmentation
HSV_S=0.7                   # Saturation
HSV_V=0.4                   # Value
DEGREES=0.0                 # Rotation
TRANSLATE=0.1               # Translation
SCALE=0.5                   # Scale
SHEAR=0.0                   # Shear
FLIPUD=0.0                  # Flip up-down
FLIPLR=0.5                  # Flip left-right
MOSAIC=1.0                  # Mosaic augmentation

# Training
PATIENCE=50                 # Early stopping patience
CLOSE_MOSAIC=10            # Disable mosaic in last N epochs
```

## 📈 Monitoring Training

### Real-time Metrics

Monitor these key metrics during training:

- **Box Loss**: Bounding box regression quality
- **Class Loss**: Classification accuracy
- **DFL Loss**: Distribution focal loss (YOLOv10 specific)
- **mAP50**: Mean average precision at 0.5 IoU
- **mAP50-95**: mAP across IoU thresholds 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Visualizations

Check generated plots in `runs/train/yolov10_person/`:

```
results.png              → Training/validation metrics over time
confusion_matrix.png     → Classification performance
PR_curve.png            → Precision-Recall curve
F1_curve.png            → F1 score vs confidence threshold
val_batch0_labels.jpg   → Ground truth annotations
val_batch0_pred.jpg     → Model predictions
```

## 🛠️ Troubleshooting

### Common Issues

**Issue: Out of memory error**
```bash
# Solution: Reduce batch size in .env
BATCH_SIZE=16  # or lower
```

**Issue: Slow training on Mac**
```bash
# Solution: Verify MPS is active
python -c "import torch; print(torch.backends.mps.is_available())"  # Should be True

# Check .env settings
DEVICE=mps
WORKERS=0
AMP_ENABLED=false
```

**Issue: Dataset not found**
```bash
# Solution: Check paths in .env and dataset.yaml
DATASET_PATH=./datasets/coco_person  # Must match dataset.yaml
```

**Issue: Multiprocessing errors on Mac**
```bash
# Solution: Force workers to 0 in .env
WORKERS=0
```

### Performance Tuning

**Faster training (lower accuracy):**
```bash
MODEL_SIZE=n
BATCH_SIZE=32
IMAGE_SIZE=512
CLOSE_MOSAIC=0
```

**Higher accuracy (slower):**
```bash
MODEL_SIZE=l
BATCH_SIZE=16
IMAGE_SIZE=640
TRAINING_EPOCHS=200
MOSAIC=1.0
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick contribution workflow:**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Third-party licenses:**
- COCO Dataset: [COCO Terms of Use](https://cocodataset.org/#termsofuse)
- YOLOv10/Ultralytics: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@software{yolov10_person_2024,
  author = {Your Name},
  title = {YOLOv10 Person Detection Training on Apple Silicon},
  year = {2024},
  url = {https://github.com/yourusername/yolo-person}
}
```

## 📚 Resources

- [Ultralytics YOLOv10 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset Official Site](https://cocodataset.org/)
- [PyTorch MPS Backend Guide](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Silicon ML Performance](https://developer.apple.com/metal/)

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv10 implementation
- **COCO Consortium** for the comprehensive dataset
- **PyTorch Team** for MPS backend support
- **Apple** for Metal Performance Shaders optimization

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/yolo-person/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/yolo-person/discussions)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

**Made with ❤️ for the Computer Vision Community**

*Star ⭐ this repo if you find it useful!*
