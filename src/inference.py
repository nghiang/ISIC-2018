"""
Full inference pipeline:
  Stage 1: YOLOv8 → lesion bounding box
  Stage 2: MedSAM → lesion segmentation mask
  Stage 3: Attribute model → 5-channel dermoscopic attribute masks
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from yolo.config import YOLO_OUTPUT
from medsam.config import (
    MEDSAM_OUTPUT,
    ATTR_OUTPUT,
    DEVICE,
    MEDSAM_IMG_SIZE,
    ATTR_IMG_SIZE,
    ATTRIBUTES,
    MEDSAM_CHECKPOINT,
)
from medsam.models import MedSAM, AttributeSegModel


def load_yolo(weights_path=None):
    if weights_path is None:
        weights_path = str(YOLO_OUTPUT / "lesion_detect" / "weights" / "best.pt")
    return YOLO(weights_path)


def load_medsam(weights_path=None):
    if weights_path is None:
        weights_path = str(MEDSAM_OUTPUT / "medsam_best.pth")
    ckpt = MEDSAM_CHECKPOINT if Path(MEDSAM_CHECKPOINT).exists() else None
    model = MedSAM(checkpoint=ckpt).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def load_attr_model(weights_path=None):
    if weights_path is None:
        weights_path = str(ATTR_OUTPUT / "attr_best.pth")
    ckpt = MEDSAM_CHECKPOINT if Path(MEDSAM_CHECKPOINT).exists() else None
    model = AttributeSegModel(sam_checkpoint=ckpt, freeze_encoder=True).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def preprocess_for_medsam(image_np):
    """Resize and pad image to 1024×1024 for MedSAM."""
    h, w = image_np.shape[:2]
    scale = MEDSAM_IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = np.array(Image.fromarray(image_np).resize((new_w, new_h), Image.BILINEAR))

    padded = np.zeros((MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    return tensor, scale, new_h, new_w


def postprocess_mask(mask_256, original_h, original_w, resized_h, resized_w):
    """Upscale MedSAM output back to original image size."""
    mask_1024 = F.interpolate(
        mask_256,
        size=(MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    mask_1024 = mask_1024[0, 0, :resized_h, :resized_w]
    mask_orig = F.interpolate(
        mask_1024.unsqueeze(0).unsqueeze(0),
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )
    return (mask_orig[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict(image_path, yolo_model, medsam_model, attr_model):
    """
    Run full inference pipeline on a single image.

    Returns:
        bbox: (x_min, y_min, x_max, y_max) in original coords
        lesion_mask: (H, W) binary mask
        attr_masks: dict {attr_name: (H, W) binary mask}
    """
    image_np = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = image_np.shape[:2]

    # ── Stage 1: YOLOv8 lesion detection ───────────────────────────
    results = yolo_model(image_path, verbose=False)
    if len(results[0].boxes) == 0:
        print(f"No lesion detected in {image_path}")
        return None, None, None

    # Take highest-confidence detection
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()
    bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    x_min, y_min, x_max, y_max = bbox

    # ── Stage 2: MedSAM lesion segmentation ────────────────────────
    img_tensor, scale, rh, rw = preprocess_for_medsam(image_np)
    img_tensor = img_tensor.to(DEVICE)

    bbox_scaled = np.array(
        [
            x_min * scale,
            y_min * scale,
            x_max * scale,
            y_max * scale,
        ],
        dtype=np.float32,
    )
    bbox_tensor = torch.from_numpy(bbox_scaled).unsqueeze(0).to(DEVICE)

    pred_mask = medsam_model(img_tensor, bbox_tensor)
    pred_mask = torch.sigmoid(pred_mask)
    lesion_mask = postprocess_mask(pred_mask, orig_h, orig_w, rh, rw)

    # ── Stage 3: Attribute segmentation ────────────────────────────
    # Crop lesion region
    ys, xs = np.where(lesion_mask > 0)
    if len(xs) == 0:
        return bbox, lesion_mask, {}

    margin = 20
    cx_min = max(0, xs.min() - margin)
    cy_min = max(0, ys.min() - margin)
    cx_max = min(orig_w, xs.max() + margin)
    cy_max = min(orig_h, ys.max() + margin)

    crop = image_np[cy_min:cy_max, cx_min:cx_max]
    crop_resized = (
        np.array(
            Image.fromarray(crop).resize((ATTR_IMG_SIZE, ATTR_IMG_SIZE), Image.BILINEAR)
        ).astype(np.float32)
        / 255.0
    )

    crop_tensor = (
        torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    )

    # Resize to encoder input
    crop_1024 = F.interpolate(
        crop_tensor,
        size=(MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    attr_pred = attr_model(crop_1024)

    # Resize prediction back to crop size
    attr_pred = F.interpolate(
        attr_pred,
        size=(cy_max - cy_min, cx_max - cx_min),
        mode="bilinear",
        align_corners=False,
    )
    attr_pred = (torch.sigmoid(attr_pred) > 0.5).cpu().numpy().astype(np.uint8)

    # Place back into full image
    attr_masks = {}
    for i, attr_name in enumerate(ATTRIBUTES):
        full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_mask[cy_min:cy_max, cx_min:cx_max] = attr_pred[0, i]
        attr_masks[attr_name] = full_mask

    return bbox, lesion_mask, attr_masks


def save_results(image_path, bbox, lesion_mask, attr_masks, output_dir):
    """Save prediction results as PNG masks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    if lesion_mask is not None:
        Image.fromarray(lesion_mask * 255).save(output_dir / f"{stem}_segmentation.png")

    if attr_masks:
        for attr_name, mask in attr_masks.items():
            Image.fromarray(mask * 255).save(output_dir / f"{stem}_{attr_name}.png")

    print(f"Results saved for {stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full inference pipeline")
    parser.add_argument("input", type=str, help="Image path or directory of images")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--yolo-weights", type=str, default=None)
    parser.add_argument("--medsam-weights", type=str, default=None)
    parser.add_argument("--attr-weights", type=str, default=None)
    args = parser.parse_args()

    yolo_model = load_yolo(args.yolo_weights)
    medsam_model = load_medsam(args.medsam_weights)
    attr_model = load_attr_model(args.attr_weights)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    else:
        images = [input_path]

    for img_path in images:
        bbox, lesion_mask, attr_masks = predict(
            str(img_path), yolo_model, medsam_model, attr_model
        )
        if lesion_mask is not None:
            save_results(str(img_path), bbox, lesion_mask, attr_masks, args.output)
