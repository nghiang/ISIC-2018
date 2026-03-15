"""
Stage 3: Train attribute segmentation model (MedSAM encoder + multi-label decoder).
Handles the 5 dermoscopic attributes from ISIC 2018 Task 2.
  - Weighted BCE + Dice loss (to handle severe class imbalance)
  - Focal loss for rare attributes
  - AdamW with cosine annealing
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from medsam.config import (
    ATTR_DIR,
    ATTR_OUTPUT,
    MEDSAM_CHECKPOINT,
    DEVICE,
    ATTR_LR,
    ATTR_WEIGHT_DECAY,
    ATTR_EPOCHS,
    ATTR_BATCH,
    ATTR_IMG_SIZE,
    ATTRIBUTES,
    NUM_ATTRIBUTES,
    MEDSAM_IMG_SIZE,
    NUM_WORKERS,
)
from medsam.dataset import AttributeDataset
from medsam.models import AttributeSegModel
from utils.evaluate import dice_coefficient, iou_score


# ── Loss ───────────────────────────────────────────────────────────────
class FocalDiceLoss(nn.Module):
    """Weighted focal BCE + Dice loss for multi-label segmentation."""

    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if class_weights is None:
            # Based on EDA: globules 25%, milia 28.5%, neg_net 6.5%, pig_net 59%, streaks 3.5%
            freqs = np.array([0.25, 0.285, 0.065, 0.59, 0.035])
            weights = 1.0 / (freqs + 0.01)
            weights = weights / weights.sum() * len(weights)
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, pred, target):
        cw = self.class_weights.to(pred.device)

        bce_raw = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce_raw)
        focal = ((1 - pt) ** self.gamma) * bce_raw

        weighted_focal = focal * cw[None, :, None, None]
        focal_loss = weighted_focal.mean()

        pred_sig = torch.sigmoid(pred)
        smooth = 1.0
        dice_loss = 0
        for c in range(pred.shape[1]):
            intersection = (pred_sig[:, c] * target[:, c]).sum()
            union = pred_sig[:, c].sum() + target[:, c].sum()
            dice_loss += 1 - (2 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss / pred.shape[1]

        return focal_loss + dice_loss


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# ── Resize helper ──────────────────────────────────────────────────────
def resize_for_encoder(images, target_size=MEDSAM_IMG_SIZE):
    return F.interpolate(
        images, size=(target_size, target_size), mode="bilinear", align_corners=False
    )


# ── Training ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        images_1024 = resize_for_encoder(images)
        pred = model(images_1024)

        if pred.shape[-2:] != masks.shape[-2:]:
            pred = F.interpolate(
                pred, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = criterion(pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    per_class_dice = np.zeros(NUM_ATTRIBUTES)
    per_class_iou = np.zeros(NUM_ATTRIBUTES)
    n = 0

    for images, masks in tqdm(loader, desc="  val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        images_1024 = resize_for_encoder(images)
        pred = model(images_1024)

        if pred.shape[-2:] != masks.shape[-2:]:
            pred = F.interpolate(
                pred, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = criterion(pred, masks)
        total_loss += loss.item() * images.size(0)

        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        for j in range(images.size(0)):
            for c in range(NUM_ATTRIBUTES):
                per_class_dice[c] += dice_coefficient(pred_bin[j, c], masks[j, c])
                per_class_iou[c] += iou_score(pred_bin[j, c], masks[j, c])
            n += 1

    avg_loss = total_loss / len(loader.dataset)
    per_class_dice /= n
    per_class_iou /= n
    return avg_loss, per_class_dice, per_class_iou


def train(
    epochs=ATTR_EPOCHS,
    batch=ATTR_BATCH,
    lr=ATTR_LR,
    checkpoint=MEDSAM_CHECKPOINT,
    workers=NUM_WORKERS,
):
    ATTR_OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    ckpt_path = checkpoint if Path(checkpoint).exists() else None
    if ckpt_path:
        print(f"Initialising encoder from {ckpt_path}")
    else:
        print("No SAM checkpoint found — training encoder from scratch")

    if ckpt_path is None:
        model = AttributeSegModel(freeze_encoder=True)
    else:
        model = AttributeSegModel(sam_checkpoint=ckpt_path, freeze_encoder=True)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    train_ds = AttributeDataset(
        ATTR_DIR / "train", img_size=ATTR_IMG_SIZE, augment=True
    )
    val_ds = AttributeDataset(ATTR_DIR / "val", img_size=ATTR_IMG_SIZE, augment=False)
    pin_memory = torch.cuda.is_available()
    print(f"DataLoader workers: {workers}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(
        _unwrap_model(model).get_trainable_params(),
        lr=lr,
        weight_decay=ATTR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dice = 0
    epoch_bar = tqdm(range(1, epochs + 1), desc="Attr epochs")
    for epoch in epoch_bar:
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        mean_dice = val_dice.mean()
        mean_iou = val_iou.mean()
        epoch_bar.set_postfix(mean_dice=f"{mean_dice:.4f}", val_loss=f"{val_loss:.4f}")

        tqdm.write(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"mean_dice: {mean_dice:.4f} | "
            f"mean_iou: {mean_iou:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        if epoch % 10 == 0 or epoch == 1:
            for c, attr in enumerate(ATTRIBUTES):
                tqdm.write(
                    f"    {attr:25s}: dice={val_dice[c]:.4f}  iou={val_iou[c]:.4f}"
                )

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(_unwrap_model(model).state_dict(), ATTR_OUTPUT / "attr_best.pth")
            tqdm.write(f"  → saved best model (dice={best_dice:.4f})")

        if epoch % 10 == 0:
            torch.save(
                _unwrap_model(model).state_dict(),
                ATTR_OUTPUT / f"attr_epoch{epoch}.pth",
            )

    torch.save(_unwrap_model(model).state_dict(), ATTR_OUTPUT / "attr_last.pth")
    tqdm.write(f"Training complete. Best mean dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train attribute segmentation")
    parser.add_argument("--epochs", type=int, default=ATTR_EPOCHS)
    parser.add_argument("--batch", type=int, default=ATTR_BATCH)
    parser.add_argument("--lr", type=float, default=ATTR_LR)
    parser.add_argument("--checkpoint", type=str, default=MEDSAM_CHECKPOINT)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        checkpoint=args.checkpoint,
        workers=args.workers,
    )
