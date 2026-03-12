"""
Stage 1: Train YOLOv8 for lesion detection on ISIC 2018 data.
"""

import argparse
from ultralytics import YOLO

from yolo.config import (
    YOLO_DIR, YOLO_OUTPUT, YOLO_MODEL,
    YOLO_EPOCHS, YOLO_BATCH, YOLO_IMG_SIZE,
)


def train(epochs=YOLO_EPOCHS, batch=YOLO_BATCH, img_size=YOLO_IMG_SIZE, resume=False):
    dataset_yaml = str(YOLO_DIR / "dataset.yaml")
    YOLO_OUTPUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_MODEL)
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        task="detect",
        project=str(YOLO_OUTPUT),
        name="lesion_detect",
        exist_ok=True,
        resume=resume,
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    print(f"YOLOv8 training complete. Results at {YOLO_OUTPUT / 'lesion_detect'}")
    return model


def validate(weights_path: str = None):
    if weights_path is None:
        weights_path = str(YOLO_OUTPUT / "lesion_detect" / "weights" / "best.pt")
    model = YOLO(weights_path)
    results = model.val(
        data=str(YOLO_DIR / "dataset.yaml"),
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH,
    )
    print(f"mAP50: {results.box.map50:.4f}  mAP50-95: {results.box.map:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 lesion detector")
    parser.add_argument("--epochs", type=int, default=YOLO_EPOCHS)
    parser.add_argument("--batch", type=int, default=YOLO_BATCH)
    parser.add_argument("--img-size", type=int, default=YOLO_IMG_SIZE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.epochs, args.batch, args.img_size, args.resume)
