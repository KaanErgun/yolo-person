# 🚀 Getting Started

Welcome to the YOLOv10 Person Detection project!

## ⚡ Quick Setup (2 Minutes)

**Just cloned this repo? Run this:**

```bash
./first-run.sh
```

That's it! The script will set up everything automatically.

---

## 📚 What Gets Configured

The `first-run.sh` script will:

1. ✅ **Check Requirements** - Python 3.9+, disk space, GPU availability
2. ✅ **Auto-Configure** - Creates `.env` with optimal settings for your hardware
3. ✅ **Setup Environment** - Creates Python virtual environment
4. ✅ **Install Dependencies** - PyTorch, YOLOv10, OpenCV, and more
5. ✅ **Run Tests** - Verifies everything works correctly
6. ✅ **Show Next Steps** - Clear instructions for training and inference

---

## 🎯 After Setup

### Start Training

```bash
source venv/bin/activate
python train.py
```

### Run Inference

```bash
source venv/bin/activate

# Webcam
python inference.py --source 0

# Image
python inference.py --source image.jpg

# Video
python inference.py --source video.mp4
```

---

## 📖 Documentation

- **[README.md](README.md)** - Complete project documentation
- **[QUICK_START.md](QUICK_START.md)** - Detailed setup guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

---

## 🆘 Need Help?

**Test your setup:**
```bash
source venv/bin/activate
python test_setup.py
```

**View configuration:**
```bash
python config.py
```

**Edit settings:**
```bash
nano .env  # or use your preferred editor
```

---

## 🎓 What This Project Does

- 🎯 **Person Detection**: Specialized YOLOv10 model trained on COCO person subset
- 🍎 **Apple Silicon**: Optimized for M1/M2/M3/M4 chips with MPS backend
- 🔧 **Easy Config**: All settings in `.env` file
- 📊 **Complete Pipeline**: From dataset download to inference

---

## ⚙️ System Requirements

- **Python**: 3.9 or higher
- **RAM**: 16GB minimum (24GB+ recommended)
- **Storage**: 150GB for full COCO dataset
- **GPU**: Apple Silicon (M1+) or NVIDIA GPU (optional but recommended)

---

**Ready to start? Run `./first-run.sh` now!** 🎉
