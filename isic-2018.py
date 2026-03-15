#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().system('nvidia-smi')
get_ipython().system('pip install -q ultralytics')


# In[3]:


from pathlib import Path

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
DATASET_ROOT = Path("/kaggle/input/datasets/nguyenquynghia/image-dataset/dataset")


def _resolve_existing_path(candidates, glob_patterns, label):
    checked = [Path(path) for path in candidates]
    for candidate in checked:
        if candidate.exists():
            return candidate

    for pattern in glob_patterns:
        matches = sorted(KAGGLE_INPUT_ROOT.glob(pattern))
        if matches:
            return matches[0]

    checked_paths = "\n".join(f" - {path}" for path in checked)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked_paths}")


def resolve_src_dir():
    return _resolve_existing_path(
        candidates=[
            "/kaggle/input/datasets/phamtiensondeptrai1/src-training/src",
            "/kaggle/input/datasets/nguyenquynghia/src-training/src",
        ],
        glob_patterns=[
            "datasets/*/src-training/src",
        ],
        label="`src-training/src` dataset",
    )


def resolve_dataset_dir(name):
    dataset_dir = DATASET_ROOT / name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")
    return dataset_dir


print("Kaggle dataset paths ready.")
print(f"Input root -> {KAGGLE_INPUT_ROOT}")
print(f"Dataset root -> {DATASET_ROOT}")


# In[4]:


# Copy src only; read YOLO dataset directly from Kaggle input
import shutil
import sys

src_working = WORKING_DIR / "src"
if src_working.exists():
    shutil.rmtree(src_working)
shutil.copytree(resolve_src_dir(), src_working)
if str(src_working) not in sys.path:
    sys.path.insert(0, str(src_working))

from pathlib import Path
import yolo.config as yolo_config
import yolo.train as yolo_train_module

YOLO_DATASET_DIR = Path(resolve_dataset_dir("yolo"))
yolo_config.YOLO_DIR = YOLO_DATASET_DIR
yolo_train_module.YOLO_DIR = YOLO_DATASET_DIR

print(f"src -> {src_working}")
print(f"dataset/yolo read-only source -> {YOLO_DATASET_DIR}")
print(f"dataset.yaml -> {YOLO_DATASET_DIR / 'dataset.yaml'}")


# In[5]:


import yolo.train as train_yolo_module

print(f"Training YOLO from -> {train_yolo_module.YOLO_DIR}")
model = train_yolo_module.train(epochs=60, batch=32, img_size=640, workers=2)


# In[6]:


import yolo.train as validate_yolo_module

print(f"Validating YOLO from -> {validate_yolo_module.YOLO_DIR}")
validate_yolo_module.validate()


# In[7]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from ultralytics import YOLO

import yolo.config as yolo_config

weights_path = str(yolo_config.YOLO_OUTPUT / "lesion_detect" / "weights" / "best.pt")
model = YOLO(weights_path)

val_dir = yolo_config.YOLO_DIR / "images" / "val"
sample_img = sorted(val_dir.glob("*.jpg"))[0]
print(f"Sample: {sample_img.name}")
print(f"Validation source -> {val_dir}")

img = np.array(Image.open(sample_img).convert("RGB"))
height, width = img.shape[:2]

label_path = yolo_config.YOLO_DIR / "labels" / "val" / sample_img.with_suffix(".txt").name
gt_boxes = []
if label_path.exists():
    for line in label_path.read_text().strip().splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        gt_boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height))

results = model.predict(source=str(sample_img), imgsz=640, conf=0.25, verbose=False)
pred_boxes = results[0].boxes.xyxy.cpu().numpy()
pred_confs = results[0].boxes.conf.cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, title in zip(axes, ["Ground Truth", "Prediction"]):
    ax.imshow(img)
    ax.set_title(f"{sample_img.name}\n{title}")
    ax.axis("off")

for (x1, y1, x2, y2) in gt_boxes:
    axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, edgecolor="lime", facecolor="none"))
    axes[0].text(x1, y1 - 4, "lesion (GT)", color="lime", fontsize=9, bbox=dict(facecolor="black", alpha=0.4))

for (x1, y1, x2, y2), conf in zip(pred_boxes, pred_confs):
    axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, edgecolor="red", facecolor="none"))
    axes[1].text(x1, y1 - 4, f"{conf:.2f}", color="red", fontsize=9, bbox=dict(facecolor="black", alpha=0.4))

plt.tight_layout()
plt.show()
print(f"GT: {len(gt_boxes)}  Predicted: {len(pred_boxes)}")


# In[8]:


import zipfile
from pathlib import Path

output_dir = Path("/kaggle/working/outputs/yolo")
output_zip = "/kaggle/working/yolo_weights.zip"

if output_dir.exists():
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to("/kaggle/working"))
                print(f"  {f.relative_to('/kaggle/working')}")
    print(f"\nSaved → {output_zip}  (download from Kaggle Output tab)")
else:
    print("No YOLO output directory found.")


# In[9]:


import gc
import torch

for name in [
    "model", "model_yolo", "results", "result", "pred_boxes", "pred_confs",
    "img", "sample_img", "gt_boxes"
]:
    globals().pop(name, None)

collected = gc.collect()
torch.cuda.empty_cache()

print(f"Python objects collected: {collected}")


# # Training medsam

# In[10]:


get_ipython().system('pip install -q segment-anything scikit-learn tqdm')


# In[11]:


# Download SAM ViT-B checkpoint
import os, subprocess

ckpt_path = "/kaggle/working/medsam_vit_b.pth"
if not os.path.exists(ckpt_path):
    subprocess.run(
        ["wget", "-q",
         "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
         "-O", ckpt_path],
        check=True,
    )
    print(f"Downloaded → {ckpt_path}")
else:
    print(f"Checkpoint already exists: {ckpt_path}")


# In[12]:


import shutil
import sys
from pathlib import Path

src_working = WORKING_DIR / "src"
shutil.rmtree(src_working, ignore_errors=True)
shutil.copytree(resolve_src_dir(), src_working)
if str(src_working) not in sys.path:
    sys.path.insert(0, str(src_working))

import medsam.config as medsam_config
import medsam.train_segmentation as train_medsam_module

MEDSAM_DATASET_DIR = Path(resolve_dataset_dir("medsam"))
medsam_config.MEDSAM_DIR = MEDSAM_DATASET_DIR
train_medsam_module.MEDSAM_DIR = MEDSAM_DATASET_DIR

train_n = len(list((MEDSAM_DATASET_DIR / "train").glob("*.npz")))
val_n = len(list((MEDSAM_DATASET_DIR / "val").glob("*.npz")))
print(f"src -> {src_working}")
print(f"dataset/medsam read-only source -> {MEDSAM_DATASET_DIR}")
print(f"MedSAM files: train={train_n}, val={val_n}")
assert train_n > 0 and val_n > 0, "Empty MedSAM split. Check Kaggle input dataset layout."


# In[13]:


get_ipython().system('nvidia-smi')


# In[14]:


import medsam.train_segmentation as train_medsam_module

print(f"Training MedSAM from -> {train_medsam_module.MEDSAM_DIR}")
train_medsam_module.train(epochs=7, batch=2, lr=1e-4, workers=2)


# In[15]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F

import medsam.config as medsam_config
from medsam.models import MedSAM

MEDSAM_DIR = medsam_config.MEDSAM_DIR
MEDSAM_OUTPUT = medsam_config.MEDSAM_OUTPUT
MEDSAM_CHECKPOINT = medsam_config.MEDSAM_CHECKPOINT
DEVICE = medsam_config.DEVICE

weights_path = MEDSAM_OUTPUT / "medsam_best.pth"
assert weights_path.exists(), f"Missing trained weights: {weights_path}"

val_files = sorted((MEDSAM_DIR / "val").glob("*.npz"))
assert len(val_files) > 0, f"No validation npz files found in {MEDSAM_DIR / 'val'}"

sample_npz = val_files[0]
data = np.load(sample_npz)
image = data["image"].astype(np.uint8)
gt_mask = data["mask"].astype(np.uint8)
bbox = data["bbox"].astype(np.float32)

model = MedSAM(checkpoint=MEDSAM_CHECKPOINT).to(DEVICE)
state = torch.load(weights_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
bbox_t = torch.from_numpy(bbox).unsqueeze(0).float().to(DEVICE)

with torch.no_grad():
    pred_logits = model(image_t, bbox_t)
    pred_logits = F.interpolate(pred_logits, size=gt_mask.shape, mode="bilinear", align_corners=False)
    pred_prob = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()

pred_mask = (pred_prob > 0.5).astype(np.uint8)

gt_bool = gt_mask.astype(bool)
pred_bool = pred_mask.astype(bool)
intersection = np.logical_and(gt_bool, pred_bool).sum()
union = np.logical_or(gt_bool, pred_bool).sum()
dice = (2 * intersection + 1e-8) / (gt_bool.sum() + pred_bool.sum() + 1e-8)
iou = (intersection + 1e-8) / (union + 1e-8)

x1, y1, x2, y2 = bbox

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, title in zip(axes, ["Ground Truth", "Prediction"]):
    ax.imshow(image)
    ax.set_title(f"{sample_npz.stem}\n{title}")
    ax.axis("off")
    ax.add_patch(
        patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, edgecolor="yellow", facecolor="none")
    )

axes[0].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap="Greens", alpha=0.45)
axes[1].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="Reds", alpha=0.45)

plt.tight_layout()
plt.show()

print(f"Validation source -> {MEDSAM_DIR / 'val'}")
print(f"Sample: {sample_npz.stem}")
print(f"Dice: {dice:.4f}  IoU: {iou:.4f}  GT pixels: {gt_bool.sum()}  Pred pixels: {pred_bool.sum()}")


# In[16]:


import zipfile
from pathlib import Path

output_dir = Path("/kaggle/working/outputs/medsam")
output_zip = "/kaggle/working/medsam_lesion_weights.zip"

if output_dir.exists():
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to("/kaggle/working"))
                print(f"  {f.relative_to('/kaggle/working')}")
    print(f"\nSaved → {output_zip}  (download from Kaggle Output tab)")
else:
    print("No MedSAM output directory found.")


# In[17]:


import gc
import torch

for name in [
    "model", "train_medsam", "pred_logits", "pred_prob", "pred_mask",
    "image_t", "bbox_t", "gt_mask", "data", "sample_npz"
]:
    globals().pop(name, None)

collected = gc.collect()
torch.cuda.empty_cache()

print(f"Python objects collected: {collected}")


# # Training medsam attribute

# In[18]:


# Copy src only; read attribute dataset directly from Kaggle input
import shutil
import sys
from pathlib import Path

src_working = WORKING_DIR / "src"
shutil.rmtree(src_working, ignore_errors=True)
shutil.copytree(resolve_src_dir(), src_working)
if str(src_working) not in sys.path:
    sys.path.insert(0, str(src_working))

import medsam.config as medsam_config
import medsam.train_attributes as train_attr_module

ATTRIBUTE_DATASET_DIR = Path(resolve_dataset_dir("attributes"))
medsam_config.ATTR_DIR = ATTRIBUTE_DATASET_DIR
train_attr_module.ATTR_DIR = ATTRIBUTE_DATASET_DIR

train_n = len(list((ATTRIBUTE_DATASET_DIR / "train").glob("*.npz")))
val_n = len(list((ATTRIBUTE_DATASET_DIR / "val").glob("*.npz")))
print(f"src -> {src_working}")
print(f"dataset/attributes read-only source -> {ATTRIBUTE_DATASET_DIR}")
print(f"Attribute files: train={train_n}, val={val_n}")
assert train_n > 0 and val_n > 0, "Empty attribute split. Check Kaggle input dataset layout."


# In[19]:


from pathlib import Path

import medsam.config as medsam_config
import medsam.train_attributes as train_attr_module

attribute_dataset_dir = resolve_dataset_dir("attributes")
medsam_config.ATTR_DIR = Path(attribute_dataset_dir)
train_attr_module.ATTR_DIR = Path(attribute_dataset_dir)

print(f"Training attributes from -> {train_attr_module.ATTR_DIR}")
train_attr_module.train(epochs=20, batch=2, lr=1e-4, workers=2)


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from medsam.config import ATTR_DIR, ATTR_OUTPUT, ATTRIBUTES, MEDSAM_CHECKPOINT, MEDSAM_IMG_SIZE, DEVICE
from medsam.dataset import AttributeDataset
from medsam.models import AttributeSegModel

weights_path = ATTR_OUTPUT / "attr_best.pth"
assert weights_path.exists(), f"Missing trained weights: {weights_path}"

val_ds = AttributeDataset(ATTR_DIR / "val", augment=False)
assert len(val_ds) > 0, f"No validation samples found in {ATTR_DIR / 'val'}"

image_t, gt_masks_t = val_ds[0]  # image: (3,256,256), gt_masks: (5,256,256)
image_np = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
gt_masks = gt_masks_t.numpy().astype(np.uint8)

model = AttributeSegModel(sam_checkpoint=MEDSAM_CHECKPOINT, freeze_encoder=True).to(DEVICE)
state = torch.load(weights_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    x = image_t.unsqueeze(0).to(DEVICE)
    x_1024 = F.interpolate(x, size=(MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE), mode="bilinear", align_corners=False)
    pred_logits = model(x_1024)
    pred_logits = F.interpolate(pred_logits, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)
    pred_prob = torch.sigmoid(pred_logits)[0].cpu().numpy()

pred_masks = (pred_prob > 0.5).astype(np.uint8)

num_attrs = len(ATTRIBUTES)
fig, axes = plt.subplots(num_attrs, 2, figsize=(12, 3.2 * num_attrs))
if num_attrs == 1:
    axes = np.expand_dims(axes, axis=0)

dice_list, iou_list = [], []
for i, attr in enumerate(ATTRIBUTES):
    gt = gt_masks[i].astype(bool)
    pred = pred_masks[i].astype(bool)

    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    dice = (2 * inter + 1e-8) / (gt.sum() + pred.sum() + 1e-8)
    iou = (inter + 1e-8) / (union + 1e-8)
    dice_list.append(dice)
    iou_list.append(iou)

    axes[i, 0].imshow(image_np)
    axes[i, 0].imshow(np.ma.masked_where(~gt, gt), cmap="Greens", alpha=0.45)
    axes[i, 0].set_title(f"{attr} — Ground Truth")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(image_np)
    axes[i, 1].imshow(np.ma.masked_where(~pred, pred), cmap="Reds", alpha=0.45)
    axes[i, 1].set_title(f"{attr} — Pred  (Dice={dice:.3f}, IoU={iou:.3f})")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

print(f"Mean Dice: {np.mean(dice_list):.4f}  Mean IoU: {np.mean(iou_list):.4f}")


# In[21]:


import zipfile
from pathlib import Path

output_dir = Path("/kaggle/working/outputs/attributes")
output_zip = "/kaggle/working/medsam_attributes_weights.zip"

if output_dir.exists():
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to("/kaggle/working"))
                print(f"  {f.relative_to('/kaggle/working')}")
    print(f"\nSaved → {output_zip}  (download from Kaggle Output tab)")
else:
    print("No attribute output directory found.")


# In[22]:


import gc
import torch

for name in [
    "model", "train_attr", "pred_logits", "pred_prob", "pred_masks",
    "image_t", "gt_masks_t", "x", "x_1024", "val_ds"
]:
    globals().pop(name, None)

collected = gc.collect()
torch.cuda.empty_cache()

print(f"Python objects collected: {collected}")

