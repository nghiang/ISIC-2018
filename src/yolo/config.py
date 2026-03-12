from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR

TASK1_2_INPUT = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
TASK1_GT = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"

# Prepared dataset
DATASET_DIR = DATA_DIR / "dataset"
YOLO_DIR = DATASET_DIR / "yolo"

# Output
OUTPUT_DIR = BASE_DIR / "outputs"
YOLO_OUTPUT = OUTPUT_DIR / "yolo"

# ── Dataset ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ── YOLOv8 ─────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"
YOLO_EPOCHS = 100
YOLO_BATCH = 16
YOLO_IMG_SIZE = 640
