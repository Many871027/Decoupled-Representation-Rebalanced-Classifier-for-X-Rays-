import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "chest_xray"
MODELS_DIR = BASE_DIR / "artifacts"
MLRUNS_DIR = BASE_DIR / "mlruns"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters (Senior Setup VRAM constrained)
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 1 # X-Rays are grayscale, taking 1 channel saves 3x VRAM
BATCH_SIZE = 16 # Low batch size for 3GB VRAM safety

# Classes
CLASS_NAMES = ['COVID', 'NEUMONIA', 'NORMALL']
NUM_CLASSES = len(CLASS_NAMES)

# Focal Loss Parameters
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# MLFlow
MLFLOW_TRACKING_URI = f"sqlite:///{MLRUNS_DIR}/mlflow.db"
EXPERIMENT_NAME = "Chest_XRay_Senior_CNN"
