"""
Data preparation for MedSAM segmentation and attribute segmentation.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from medsam.config import (
    TASK1_2_INPUT, TASK1_GT, TASK2_GT,
    MEDSAM_DIR, ATTR_DIR,
    ATTRIBUTES, TRAIN_RATIO, RANDOM_SEED,
    MEDSAM_IMG_SIZE,
)

NUM_ATTRIBUTES = len(ATTRIBUTES)


# ── helpers ────────────────────────────────────────────────────────────
def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resize_longest_side(img: np.ndarray, target: int):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil), scale


def pad_to_square(img: np.ndarray, target: int):
    h, w = img.shape[:2]
    if img.ndim == 3:
        padded = np.zeros((target, target, img.shape[2]), dtype=img.dtype)
    else:
        padded = np.zeros((target, target), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


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


# ── MedSAM data preparation ───────────────────────────────────────────
def prepare_medsam(train_ids, val_ids):
    """Save MedSAM-ready npz files: image_1024, mask_1024, bbox_1024."""
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        out_dir = MEDSAM_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            npz_path = out_dir / f"{img_id}.npz"
            if npz_path.exists():
                continue

            img = np.array(Image.open(TASK1_2_INPUT / f"{img_id}.jpg").convert("RGB"))
            mask = np.array(Image.open(TASK1_GT / f"{img_id}_segmentation.png"))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 127).astype(np.uint8)

            img_resized, scale = resize_longest_side(img, MEDSAM_IMG_SIZE)
            mask_resized, _ = resize_longest_side(mask * 255, MEDSAM_IMG_SIZE)
            mask_resized = (mask_resized > 127).astype(np.uint8)

            img_padded = pad_to_square(img_resized, MEDSAM_IMG_SIZE)
            mask_padded = pad_to_square(mask_resized, MEDSAM_IMG_SIZE)

            bbox = mask_to_bbox(mask_padded)
            if bbox is None:
                continue

            np.savez_compressed(
                npz_path,
                image=img_padded,
                mask=mask_padded,
                bbox=np.array(bbox, dtype=np.float32),
            )

    print(f"MedSAM dataset ready at {MEDSAM_DIR}  ({len(train_ids)} train, {len(val_ids)} val)")


# ── Attribute segmentation data preparation ────────────────────────────
def prepare_attributes(train_ids, val_ids):
    """Save attribute data: image + 5-channel attribute mask + lesion mask."""
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        out_dir = ATTR_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            npz_path = out_dir / f"{img_id}.npz"
            if npz_path.exists():
                continue

            img = np.array(Image.open(TASK1_2_INPUT / f"{img_id}.jpg").convert("RGB"))
            seg_mask = np.array(Image.open(TASK1_GT / f"{img_id}_segmentation.png"))
            if seg_mask.ndim == 3:
                seg_mask = seg_mask[:, :, 0]
            seg_mask = (seg_mask > 127).astype(np.uint8)

            h, w = img.shape[:2]
            attr_masks = np.zeros((NUM_ATTRIBUTES, h, w), dtype=np.uint8)
            for i, attr in enumerate(ATTRIBUTES):
                attr_path = TASK2_GT / f"{img_id}_attribute_{attr}.png"
                if attr_path.exists():
                    a = np.array(Image.open(attr_path))
                    if a.ndim == 3:
                        a = a[:, :, 0]
                    attr_masks[i] = (a > 127).astype(np.uint8)

            np.savez_compressed(
                npz_path,
                image=img,
                seg_mask=seg_mask,
                attr_masks=attr_masks,
            )

    print(f"Attribute dataset ready at {ATTR_DIR}  ({len(train_ids)} train, {len(val_ids)} val)")


def main():
    image_ids = get_image_ids()
    print(f"Found {len(image_ids)} images with segmentation masks")
    train_ids, val_ids = split_ids(image_ids)
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    prepare_medsam(train_ids, val_ids)
    prepare_attributes(train_ids, val_ids)


if __name__ == "__main__":
    main()
