"""
Evaluation metrics for segmentation tasks.
"""

import torch
import numpy as np


def dice_coefficient(pred, target, smooth=1.0):
    """Compute Dice coefficient between two binary tensors or arrays."""
    if isinstance(pred, torch.Tensor):
        pred = pred.float()
        target = target.float()
        intersection = (pred * target).sum()
        return (
            (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        ).item()
    else:
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        intersection = (pred * target).sum()
        return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    """Compute IoU (Jaccard index) between two binary tensors or arrays."""
    if isinstance(pred, torch.Tensor):
        pred = pred.float()
        target = target.float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return ((intersection + smooth) / (union + smooth)).item()
    else:
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    if isinstance(pred, torch.Tensor):
        correct = (pred == target).float().sum()
        total = torch.numel(pred)
        return (correct / total).item()
    else:
        return (pred == target).sum() / pred.size
