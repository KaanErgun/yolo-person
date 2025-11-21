"""
YOLOv10 Inference Script for Person Detection
Run inference on images, videos, or webcam with .env configuration
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
from config import Config


def run_inference(source=None, model_path=None, conf=None, show=False):
    """
    Run inference with trained YOLOv10 model using configuration from .env
    
    Args:
        source: Input source (overrides .env if provided)
        model_path: Path to trained model (overrides .env if provided)
        conf: Confidence threshold (overrides .env if provided)
        show: Show results in window
    """
    # Get configuration
    cfg = Config
    
    # Use command line arguments if provided, otherwise use config
    if model_path is None:
        model_path = f'{cfg.PROJECT_NAME}/{cfg.EXPERIMENT_NAME}/weights/best.pt'
    if source is None:
        source = cfg.INFERENCE_SOURCE
    if conf is None:
        conf = cfg.CONFIDENCE_THRESHOLD
    
    print("="*60)
    print("YOLOv10 Inference Configuration")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence: {conf}")
    print(f"IOU: {cfg.IOU_THRESHOLD}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print("="*60 + "\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        iou=cfg.IOU_THRESHOLD,
        imgsz=cfg.IMAGE_SIZE,
        save=cfg.SAVE_RESULTS,
        show=show,
        project=cfg.INFERENCE_OUTPUT,
        name=cfg.EXPERIMENT_NAME,
        exist_ok=True
    )
    
    print("\n" + "="*60)
    print("✅ Inference complete!")
    print("="*60)
    if cfg.SAVE_RESULTS:
        print(f"Results saved to: {cfg.INFERENCE_OUTPUT}/{cfg.EXPERIMENT_NAME}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLOv10 Person Detection Inference')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (overrides .env)')
    parser.add_argument('--source', type=str, default=None,
                        help='Input source: image path, video path, directory, or webcam number (overrides .env)')
    parser.add_argument('--conf', type=float, default=None,
                        help='Confidence threshold (overrides .env)')
    parser.add_argument('--show', action='store_true',
                        help='Show results in window')
    
    args = parser.parse_args()
    
    run_inference(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        show=args.show
    )
