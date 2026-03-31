"""
Central configuration for Bone Fracture Classifier.
All paths, hyperparameters and constants live here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
TEST_DIR   = os.path.join(DATA_DIR, "test")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "pkdarabi/bone-fracture-detection-computer-vision-project"
CLASS_NAMES    = ["Fractured", "Not Fractured"]
NUM_CLASSES    = 2
IMG_SIZE       = 224          # EfficientNetB0 default

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 1e-4
LR_STEP     = 5               # StepLR: drop every N epochs
LR_GAMMA    = 0.5
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 4
SEED        = 42
PATIENCE    = 5               # early-stop patience

# ─── Model ────────────────────────────────────────────────────────────────────
BACKBONE       = "efficientnet_b0"
PRETRAINED     = True
DROPOUT        = 0.4

# ─── Augmentation ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (X-rays are converted to RGB)
STD  = [0.229, 0.224, 0.225]

# ─── Flask app ────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 5000
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB upload limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
