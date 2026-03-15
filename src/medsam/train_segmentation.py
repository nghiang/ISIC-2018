"""
Stage 2: Fine-tune MedSAM for lesion segmentation on ISIC 2018.
  - Freeze prompt encoder
  - Fine-tune image encoder + mask decoder
  - BCE + Dice loss
  - AdamW optimizer with cosine annealing
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from medsam.config import (
    MEDSAM_DIR,
    MEDSAM_OUTPUT,
    MEDSAM_CHECKPOINT,
    DEVICE,
    MEDSAM_LR,
    MEDSAM_WEIGHT_DECAY,
    MEDSAM_EPOCHS,
    MEDSAM_BATCH,
    NUM_WORKERS,
)
from medsam.dataset import MedSAMDataset
from medsam.models import MedSAM
from utils.evaluate import dice_coefficient, iou_score


# ── Loss ───────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# ── Training ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for images, bboxes, masks in pbar:
        images = images.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)

        pred_masks = model(images, bboxes)

        masks_resized = F.interpolate(masks, size=pred_masks.shape[-2:], mode="nearest")

        loss = criterion(pred_masks, masks_resized)
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
    total_dice = 0
    total_iou = 0
    n = 0
    for images, bboxes, masks in tqdm(loader, desc="  val", leave=False):
        images = images.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)

        pred_masks = model(images, bboxes)
        masks_resized = F.interpolate(masks, size=pred_masks.shape[-2:], mode="nearest")

        loss = criterion(pred_masks, masks_resized)
        total_loss += loss.item() * images.size(0)

        pred_bin = (torch.sigmoid(pred_masks) > 0.5).float()
        for j in range(images.size(0)):
            total_dice += dice_coefficient(pred_bin[j, 0], masks_resized[j, 0])
            total_iou += iou_score(pred_bin[j, 0], masks_resized[j, 0])
            n += 1

    return total_loss / len(loader.dataset), total_dice / n, total_iou / n


def train(
    epochs=MEDSAM_EPOCHS,
    batch=MEDSAM_BATCH,
    lr=MEDSAM_LR,
    checkpoint=MEDSAM_CHECKPOINT,
    workers=NUM_WORKERS,
):
    MEDSAM_OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading MedSAM from {checkpoint}")

    ckpt_path = checkpoint if Path(checkpoint).exists() else None
    if ckpt_path is None:
        model = MedSAM()
    else:
        model = MedSAM(checkpoint=ckpt_path)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    train_ds = MedSAMDataset(MEDSAM_DIR / "train", augment=True)
    val_ds = MedSAMDataset(MEDSAM_DIR / "val", augment=False)
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

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        _unwrap_model(model).get_trainable_params(),
        lr=lr,
        weight_decay=MEDSAM_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dice = 0
    epoch_bar = tqdm(range(1, epochs + 1), desc="MedSAM epochs")
    for epoch in epoch_bar:
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0
        epoch_bar.set_postfix(val_dice=f"{val_dice:.4f}", val_loss=f"{val_loss:.4f}")

        tqdm.write(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_dice: {val_dice:.4f} | "
            f"val_iou: {val_iou:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                _unwrap_model(model).state_dict(), MEDSAM_OUTPUT / "medsam_best.pth"
            )
            tqdm.write(f"  → saved best model (dice={best_dice:.4f})")

        if epoch % 10 == 0:
            torch.save(
                _unwrap_model(model).state_dict(),
                MEDSAM_OUTPUT / f"medsam_epoch{epoch}.pth",
            )

    torch.save(_unwrap_model(model).state_dict(), MEDSAM_OUTPUT / "medsam_last.pth")
    tqdm.write(f"Training complete. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedSAM for lesion segmentation")
    parser.add_argument("--epochs", type=int, default=MEDSAM_EPOCHS)
    parser.add_argument("--batch", type=int, default=MEDSAM_BATCH)
    parser.add_argument("--lr", type=float, default=MEDSAM_LR)
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
