from pathlib import Path
import torch

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR

TASK1_2_INPUT = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
TASK1_GT = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"
TASK2_GT = DATA_DIR / "ISIC2018_Task2_Training_GroundTruth_v3"

# Prepared dataset directories
DATASET_DIR = DATA_DIR / "dataset"
MEDSAM_DIR = DATASET_DIR / "medsam"
ATTR_DIR = DATASET_DIR / "attributes"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
MEDSAM_OUTPUT = OUTPUT_DIR / "medsam"
ATTR_OUTPUT = OUTPUT_DIR / "attributes"

# ── Dataset ────────────────────────────────────────────────────────────
ATTRIBUTES = [
    "globules",
    "milia_like_cyst",
    "negative_network",
    "pigment_network",
    "streaks",
]
NUM_ATTRIBUTES = len(ATTRIBUTES)
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ── Preprocessing ──────────────────────────────────────────────────────
MEDSAM_IMG_SIZE = 1024
ATTR_IMG_SIZE = 256

# ── MedSAM ─────────────────────────────────────────────────────────────
MEDSAM_CHECKPOINT = "medsam_vit_b.pth"
SAM_MODEL_TYPE = "vit_b"
MEDSAM_LR = 1e-4
MEDSAM_WEIGHT_DECAY = 0.01
MEDSAM_EPOCHS = 50
MEDSAM_BATCH = 4
BBOX_SHIFT = 20  # random perturbation for bbox prompts during training
CROP_MARGIN = 50  # pixel margin around bbox when cropping to lesion region

# ── Attribute Segmentation ─────────────────────────────────────────────
ATTR_LR = 1e-4
ATTR_WEIGHT_DECAY = 0.01
ATTR_EPOCHS = 80
ATTR_BATCH = 4

# ── Device ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
