"""
YOLOv10 Training Script for Person Detection
Trains YOLOv10 model on COCO Person dataset with .env configuration
"""
from ultralytics import YOLO
import torch
from config import Config


def train_yolov10():
    """
    Train YOLOv10 model using configuration from .env file
    """
    # Get configuration
    cfg = Config
    
    # Determine device
    device = cfg.DEVICE.lower()
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            device = '0'  # NVIDIA GPU
        else:
            device = 'cpu'
    
    print("="*60)
    print("YOLOv10 Training Configuration")
    print("="*60)
    print(f"Model Size: YOLOv10{cfg.MODEL_SIZE}")
    print(f"Dataset: {cfg.DATASET_YAML}")
    print(f"Epochs: {cfg.EPOCHS}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Device: {device}")
    print(f"Workers: {cfg.WORKERS}")
    print(f"Cache: {cfg.CACHE}")
    print(f"AMP: {cfg.AMP}")
    print(f"Optimizer: {cfg.OPTIMIZER}")
    print("="*60 + "\n")
    
    # Load model
    if cfg.RESUME:
        print("Resuming training from last checkpoint...")
        model = YOLO(f'{cfg.PROJECT_NAME}/{cfg.EXPERIMENT_NAME}/weights/last.pt')
    else:
        model_name = f'yolov10{cfg.MODEL_SIZE}.pt' if cfg.USE_PRETRAINED else f'yolov10{cfg.MODEL_SIZE}.yaml'
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=cfg.DATASET_YAML,
        epochs=cfg.EPOCHS,
        batch=cfg.BATCH_SIZE,
        imgsz=cfg.IMAGE_SIZE,
        device=device,
        project=cfg.PROJECT_NAME,
        name=cfg.EXPERIMENT_NAME,
        patience=cfg.PATIENCE,
        save_period=cfg.SAVE_PERIOD,
        workers=cfg.WORKERS,
        exist_ok=True,
        resume=cfg.RESUME,
        pretrained=cfg.USE_PRETRAINED,
        cache=cfg.CACHE,
        amp=cfg.AMP,
        close_mosaic=cfg.CLOSE_MOSAIC,
        rect=True,  # Rectangular training for faster validation
        # Optimization settings
        optimizer=cfg.OPTIMIZER,
        lr0=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        dropout=cfg.DROPOUT,
        label_smoothing=cfg.LABEL_SMOOTHING,
        # Augmentation
        hsv_h=cfg.HSV_H,
        hsv_s=cfg.HSV_S,
        hsv_v=cfg.HSV_V,
        degrees=cfg.DEGREES,
        translate=cfg.TRANSLATE,
        scale=cfg.SCALE,
        shear=cfg.SHEAR,
        flipud=cfg.FLIPUD,
        fliplr=cfg.FLIPLR,
        mosaic=cfg.MOSAIC,
    )
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print(f"Results saved to: {cfg.PROJECT_NAME}/{cfg.EXPERIMENT_NAME}")
    print(f"Best model: {cfg.PROJECT_NAME}/{cfg.EXPERIMENT_NAME}/weights/best.pt")
    print(f"Last model: {cfg.PROJECT_NAME}/{cfg.EXPERIMENT_NAME}/weights/last.pt")
    
    # Validate the best model
    print("\n" + "="*60)
    print("Validating best model...")
    print("="*60)
    best_model = YOLO(f'{cfg.PROJECT_NAME}/{cfg.EXPERIMENT_NAME}/weights/best.pt')
    metrics = best_model.val(data=cfg.DATASET_YAML)
    
    print(f"\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return results, metrics


if __name__ == "__main__":
    train_yolov10()
