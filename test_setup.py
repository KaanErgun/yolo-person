#!/usr/bin/env python3
"""
Quick test script to verify YOLOv10 setup
Tests configuration, dataset, and model loading
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing imports...")
    try:
        import torch
        import torchvision
        from ultralytics import YOLO
        import cv2
        import numpy as np
        from config import Config
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pytorch():
    """Test PyTorch and MPS availability"""
    print("\n🔍 Testing PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ MPS available: {torch.backends.mps.is_available()}")
        print(f"✅ MPS built: {torch.backends.mps.is_built()}")
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    try:
        from config import Config
        Config.validate()
        print(f"✅ Model: YOLOv10{Config.MODEL_SIZE}")
        print(f"✅ Device: {Config.DEVICE}")
        print(f"✅ Dataset: {Config.DATASET_YAML}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dataset():
    """Test if dataset exists"""
    print("\n🔍 Testing dataset...")
    try:
        from config import Config
        dataset_path = Path(Config.PERSON_DATASET_PATH)
        train_images = dataset_path / "images" / "train"
        train_labels = dataset_path / "labels" / "train"
        val_images = dataset_path / "images" / "val"
        val_labels = dataset_path / "labels" / "val"
        
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        train_img_count = len(list(train_images.glob("*.jpg"))) if train_images.exists() else 0
        val_img_count = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0
        
        print(f"✅ Training images: {train_img_count:,}")
        print(f"✅ Validation images: {val_img_count:,}")
        
        if train_img_count == 0 or val_img_count == 0:
            print("⚠️  Dataset exists but images not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_model_loading():
    """Test if YOLOv10 model can be loaded"""
    print("\n🔍 Testing model loading...")
    try:
        from ultralytics import YOLO
        from config import Config
        
        model_name = f"yolov10{Config.MODEL_SIZE}.pt"
        print(f"   Loading {model_name}...")
        model = YOLO(model_name)
        print(f"✅ Model loaded successfully")
        print(f"   Model will be downloaded if not present")
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("YOLOv10 Person Detection - System Test")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Configuration", test_config()))
    results.append(("Dataset", test_dataset()))
    results.append(("Model Loading", test_model_loading()))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train!")
        print("\nNext steps:")
        print("  1. python train.py          # Start training")
        print("  2. python inference.py      # Run inference after training")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
