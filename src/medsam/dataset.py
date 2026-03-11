"""
PyTorch datasets for MedSAM lesion segmentation and attribute segmentation.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from medsam.config import BBOX_SHIFT, MEDSAM_IMG_SIZE, ATTR_IMG_SIZE, ATTRIBUTES


# ── MedSAM Dataset ─────────────────────────────────────────────────────
class MedSAMDataset(Dataset):
    """Loads pre-processed npz files for MedSAM training."""

    def __init__(self, data_dir: Path, augment: bool = False):
        self.files = sorted(data_dir.glob("*.npz"))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _perturb_bbox(self, bbox, img_size):
        shift = np.random.randint(-BBOX_SHIFT, BBOX_SHIFT + 1, size=4)
        x_min = max(0, bbox[0] + shift[0])
        y_min = max(0, bbox[1] + shift[1])
        x_max = min(img_size - 1, bbox[2] + shift[2])
        y_max = min(img_size - 1, bbox[3] + shift[3])
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data["image"].astype(np.float32)  # (H, W, 3)
        mask = data["mask"].astype(np.float32)     # (H, W)
        bbox = data["bbox"].astype(np.float32)     # (4,)

        image = image / 255.0

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
                w = image.shape[1]
                x_min, y_min, x_max, y_max = bbox
                bbox = np.array([w - x_max, y_min, w - x_min, y_max], dtype=np.float32)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
                h = image.shape[0]
                x_min, y_min, x_max, y_max = bbox
                bbox = np.array([x_min, h - y_max, x_max, h - y_min], dtype=np.float32)
            bbox = self._perturb_bbox(bbox, MEDSAM_IMG_SIZE)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        bbox = torch.from_numpy(bbox)

        return image, bbox, mask


# ── Attribute Segmentation Dataset ─────────────────────────────────────
class AttributeDataset(Dataset):
    """Loads pre-processed npz files for attribute segmentation.
    Crops the lesion region and resizes to ATTR_IMG_SIZE.
    """

    def __init__(self, data_dir: Path, img_size: int = ATTR_IMG_SIZE, augment: bool = False):
        self.files = sorted(data_dir.glob("*.npz"))
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _crop_to_lesion(self, image, seg_mask, attr_masks, margin=20):
        ys, xs = np.where(seg_mask > 0)
        if len(xs) == 0:
            return image, attr_masks

        h, w = image.shape[:2]
        x_min = max(0, xs.min() - margin)
        y_min = max(0, ys.min() - margin)
        x_max = min(w, xs.max() + margin)
        y_max = min(h, ys.max() + margin)

        image_crop = image[y_min:y_max, x_min:x_max]
        attr_crop = attr_masks[:, y_min:y_max, x_min:x_max]
        return image_crop, attr_crop

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data["image"]
        seg_mask = data["seg_mask"]
        attr_masks = data["attr_masks"]

        image, attr_masks = self._crop_to_lesion(image, seg_mask, attr_masks)

        pil_img = Image.fromarray(image).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        image = np.array(pil_img, dtype=np.float32) / 255.0

        resized_attrs = np.zeros(
            (len(ATTRIBUTES), self.img_size, self.img_size), dtype=np.float32
        )
        for i in range(attr_masks.shape[0]):
            m = Image.fromarray(attr_masks[i] * 255).resize(
                (self.img_size, self.img_size), Image.NEAREST
            )
            resized_attrs[i] = (np.array(m) > 127).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                resized_attrs = np.flip(resized_attrs, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0).copy()
                resized_attrs = np.flip(resized_attrs, axis=1).copy()
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-0.1, 0.1)
                image = np.clip(image * alpha + beta, 0, 1)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        masks = torch.from_numpy(resized_attrs).float()

        return image, masks
