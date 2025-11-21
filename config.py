"""
Configuration loader for YOLOv10 Person Detection
Loads settings from .env file
"""
import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for YOLOv10 Person Detection"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    # Dataset Configuration
    COCO_DATASET_PATH = os.getenv('COCO_DATASET_PATH', './data/coco')
    PERSON_DATASET_PATH = os.getenv('PERSON_DATASET_PATH', './datasets/coco_person')
    DATASET_YAML = os.getenv('DATASET_YAML', 'dataset.yaml')
    
    # Model Configuration
    MODEL_SIZE = os.getenv('MODEL_SIZE', 's')
    USE_PRETRAINED = os.getenv('USE_PRETRAINED', 'true').lower() == 'true'
    
    # Training Configuration
    EPOCHS = int(os.getenv('EPOCHS', '50'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '24'))
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '640'))
    DEVICE = os.getenv('DEVICE', 'mps')
    WORKERS = int(os.getenv('WORKERS', '0'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    
    # Output Configuration
    PROJECT_DIR = os.getenv('PROJECT_DIR', 'runs/train')
    PROJECT_NAME = PROJECT_DIR  # Alias for compatibility
    EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'yolov10_person')
    SAVE_PERIOD = int(os.getenv('SAVE_PERIOD', '10'))
    PATIENCE = int(os.getenv('PATIENCE', '50'))
    RESUME = os.getenv('RESUME', 'false').lower() == 'true'
    
    # Inference Configuration
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))
    MAX_DETECTIONS = int(os.getenv('MAX_DETECTIONS', '300'))
    SHOW_RESULTS = os.getenv('SHOW_RESULTS', 'true').lower() == 'true'
    SAVE_RESULTS = os.getenv('SAVE_RESULTS', 'true').lower() == 'true'
    INFERENCE_OUTPUT_DIR = os.getenv('INFERENCE_OUTPUT_DIR', 'runs/detect')
    INFERENCE_OUTPUT = INFERENCE_OUTPUT_DIR  # Alias for compatibility
    INFERENCE_SOURCE = os.getenv('VIDEO_SOURCE', '0')  # Default source
    
    # Video Inference Configuration
    VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', '0')
    SAVE_VIDEO = os.getenv('SAVE_VIDEO', 'true').lower() == 'true'
    SHOW_FPS = os.getenv('SHOW_FPS', 'true').lower() == 'true'
    
    # Advanced Training Configuration
    AMP = os.getenv('AMP', 'false').lower() == 'true'
    CACHE = os.getenv('CACHE', 'false')
    if CACHE.lower() == 'true':
        CACHE = True
    elif CACHE.lower() == 'false':
        CACHE = False
    # else keep as string for 'ram' option
    
    OPTIMIZER = os.getenv('OPTIMIZER', 'AdamW')
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '0.0005'))
    WARMUP_EPOCHS = float(os.getenv('WARMUP_EPOCHS', '3.0'))
    MOMENTUM = float(os.getenv('MOMENTUM', '0.937'))
    DROPOUT = float(os.getenv('DROPOUT', '0.0'))
    LABEL_SMOOTHING = float(os.getenv('LABEL_SMOOTHING', '0.0'))
    CLOSE_MOSAIC = int(os.getenv('CLOSE_MOSAIC', '10'))
    
    # Data Augmentation
    HSV_H = float(os.getenv('HSV_H', '0.015'))
    HSV_S = float(os.getenv('HSV_S', '0.7'))
    HSV_V = float(os.getenv('HSV_V', '0.4'))
    DEGREES = float(os.getenv('DEGREES', '0.0'))
    TRANSLATE = float(os.getenv('TRANSLATE', '0.1'))
    SCALE = float(os.getenv('SCALE', '0.5'))
    SHEAR = float(os.getenv('SHEAR', '0.0'))
    FLIPUD = float(os.getenv('FLIPUD', '0.0'))
    FLIPLR = float(os.getenv('FLIPLR', '0.5'))
    MOSAIC = float(os.getenv('MOSAIC', '1.0'))
    
    # Hardware Configuration
    GPU_MEMORY_LIMIT = os.getenv('GPU_MEMORY_LIMIT', '')
    DETERMINISTIC = os.getenv('DETERMINISTIC', 'true').lower() == 'true'
    
    # Logging Configuration
    VERBOSE = os.getenv('VERBOSE', 'true').lower() == 'true'
    TENSORBOARD = os.getenv('TENSORBOARD', 'false').lower() == 'true'
    WANDB = os.getenv('WANDB', 'false').lower() == 'true'
    WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'yolov10-person-detection')
    
    @classmethod
    def get_model_path(cls, model_size: str = None) -> str:
        """Get model file path"""
        size = model_size or cls.MODEL_SIZE
        return f'yolov10{size}.pt'
    
    @classmethod
    def get_absolute_path(cls, path: Union[str, Path]) -> Path:
        """Convert relative path to absolute path"""
        path = Path(path)
        if path.is_absolute():
            return path
        return cls.PROJECT_ROOT / path
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("Configuration Settings")
        print("="*60)
        print(f"Model Size: YOLOv10{cls.MODEL_SIZE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Workers: {cls.WORKERS}")
        print(f"Use Pretrained: {cls.USE_PRETRAINED}")
        print(f"Dataset: {cls.DATASET_YAML}")
        print("="*60)
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        # Validate model size
        if cls.MODEL_SIZE not in ['n', 's', 'm', 'l', 'x']:
            errors.append(f"Invalid MODEL_SIZE: {cls.MODEL_SIZE}. Must be one of: n, s, m, l, x")
        
        # Validate device
        if cls.DEVICE not in ['cpu', 'mps', 'cuda'] and not cls.DEVICE.isdigit():
            errors.append(f"Invalid DEVICE: {cls.DEVICE}. Must be: cpu, mps, cuda, or GPU id")
        
        # Validate numeric ranges
        if cls.BATCH_SIZE <= 0:
            errors.append(f"BATCH_SIZE must be positive, got: {cls.BATCH_SIZE}")
        
        if cls.EPOCHS <= 0:
            errors.append(f"EPOCHS must be positive, got: {cls.EPOCHS}")
        
        if not 0 <= cls.CONFIDENCE_THRESHOLD <= 1:
            errors.append(f"CONFIDENCE_THRESHOLD must be between 0 and 1, got: {cls.CONFIDENCE_THRESHOLD}")
        
        if not 0 <= cls.IOU_THRESHOLD <= 1:
            errors.append(f"IOU_THRESHOLD must be between 0 and 1, got: {cls.IOU_THRESHOLD}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  {e}")
    print("\nPlease check your .env file and fix the errors.")


# Convenience function to get config instance
def get_config() -> Config:
    """Get configuration instance"""
    return Config


if __name__ == '__main__':
    # Print configuration when run directly
    Config.print_config()
