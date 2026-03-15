"""
Data preparation for YOLOv8 lesion detection.
Converts ISIC segmentation masks to YOLO bounding-box format.
"""

import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from yolo.config import (
    TASK1_2_INPUT,
    TASK1_GT,
    YOLO_DIR,
    TRAIN_RATIO,
    RANDOM_SEED,
)


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h


def get_image_ids():
    img_ids = []
    for p in sorted(TASK1_2_INPUT.glob("*.jpg")):
        img_id = p.stem
        seg = TASK1_GT / f"{img_id}_segmentation.png"
        if seg.exists():
            img_ids.append(img_id)
    return img_ids


def split_ids(image_ids):
    return train_test_split(
        image_ids, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, shuffle=True
    )


def prepare_yolo(train_ids, val_ids):
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            src_img = TASK1_2_INPUT / f"{img_id}.jpg"
            dst_img = img_dir / f"{img_id}.jpg"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            mask = np.array(Image.open(TASK1_GT / f"{img_id}_segmentation.png"))
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            h, w = mask.shape[:2]
            cx, cy, bw, bh = bbox_to_yolo(bbox, w, h)
            lbl_path = lbl_dir / f"{img_id}.txt"
            with open(lbl_path, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: lesion\n"
        "task: detect\n"
    )
    print(
        f"YOLO dataset ready at {YOLO_DIR}  ({len(train_ids)} train, {len(val_ids)} val)"
    )


def main():
    image_ids = get_image_ids()
    print(f"Found {len(image_ids)} images with segmentation masks")
    train_ids, val_ids = split_ids(image_ids)
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    prepare_yolo(train_ids, val_ids)


if __name__ == "__main__":
    main()
