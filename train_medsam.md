# Training MedSAM for Lesion Segmentation

## 1. Overview

MedSAM (Medical Segment Anything Model) is a medical adaptation of the Segment Anything Model (SAM).
Instead of training a segmentation model from scratch, MedSAM **fine-tunes the SAM architecture on medical imaging datasets**.

The training strategy adapts the original SAM components for medical segmentation tasks.

MedSAM consists of three main modules:

```
Image Encoder
Prompt Encoder
Mask Decoder
```

During training:

* the **prompt encoder is frozen**
* the **image encoder and mask decoder are fine-tuned**

This allows the model to retain the strong visual representation learned from large-scale datasets while adapting to medical image characteristics.

---

# 2. Training Objective

The goal of training MedSAM is to learn to generate segmentation masks conditioned on prompts.

For lesion segmentation, the training inputs are:

```
Input Image
Bounding Box Prompt
Ground Truth Segmentation Mask
```

The model learns to predict the segmentation mask corresponding to the object specified by the prompt.

---

# 3. Dataset Preparation

MedSAM training requires:

```
image
segmentation mask
bounding box prompt
```

For lesion segmentation tasks (such as ISIC):

```
0 → background
1 → lesion
```

Dataset structure:

```
dataset/
    images/
        ISIC_0001.jpg
        ISIC_0002.jpg

    masks/
        ISIC_0001.png
        ISIC_0002.png
```

The bounding box prompt is generated automatically from the segmentation mask.

---

# 4. Data Preprocessing

Medical images typically vary significantly in resolution and intensity distribution.
Therefore, preprocessing is required before training.

## 4.1 Image Resizing

Images are resized such that the longest side equals:

```
1024 pixels
```

The aspect ratio is preserved.

If necessary, padding is applied to obtain:

```
1024 × 1024
```

---

## 4.2 Intensity Normalization

Images are normalized using min–max normalization:

```
I_norm = (I - I_min) / (I_max - I_min)
```

This improves model stability during training.

---

## 4.3 Bounding Box Generation

Bounding boxes are generated from ground truth masks.

For each segmentation mask:

```
x_min = min(x where mask > 0)
y_min = min(y where mask > 0)
x_max = max(x where mask > 0)
y_max = max(y where mask > 0)
```

To improve robustness, random perturbation can be added:

```
bbox = bbox + random_noise
```

This prevents the model from overfitting to perfectly aligned prompts.

---

# 5. Training Dataset Loader

The training pipeline uses a custom dataset loader.

Each training sample contains:

```
image tensor
bounding box prompt
ground truth mask
```

During training:

1. Load image
2. Load corresponding mask
3. Generate bounding box
4. Apply augmentation
5. Convert to tensor

Output format:

```
(image, bbox_prompt, gt_mask)
```

---

# 6. Model Architecture

MedSAM adapts the original SAM architecture.

### Image Encoder

The image encoder is a Vision Transformer (ViT).

It converts the input image into a feature embedding:

```
Image → ViT Encoder → Image Embedding
```

---

### Prompt Encoder

The prompt encoder converts prompts into embeddings.

In this pipeline the prompt is a bounding box:

```
bbox → prompt embedding
```

The prompt encoder remains **frozen during training**.

---

### Mask Decoder

The mask decoder combines image and prompt embeddings to generate the segmentation mask.

```
image embedding
+
prompt embedding
        ↓
mask decoder
        ↓
segmentation mask
```

---

# 7. Loss Function

MedSAM training uses a combination of:

### Binary Cross Entropy (BCE)

```
L_BCE = -[y log(p) + (1-y) log(1-p)]
```

### Dice Loss

Dice loss improves segmentation of small regions.

```
Dice = 2|X ∩ Y| / (|X| + |Y|)
```

Final loss:

```
L_total = L_BCE + L_Dice
```

This combination balances pixel-wise accuracy and region overlap.

---

# 8. Training Configuration

Typical training configuration:

```
Optimizer: AdamW
Learning rate: 1e-4
Batch size: 8 – 32
Epochs: 50 – 100
Weight decay: 0.01
```

Learning rate scheduling can be applied:

```
Cosine Annealing
or
Step Decay
```

---

# 9. Training Procedure

The training loop follows these steps:

```
for each batch:

    image → image encoder

    bbox → prompt encoder

    combine embeddings

    mask decoder → predicted mask

    compute BCE + Dice loss

    update parameters
```

Only the **image encoder and mask decoder parameters are updated**.

---

# 10. Data Augmentation

To improve generalization, the following augmentations are applied:

Geometric augmentations:

```
horizontal flip
vertical flip
random rotation
random scaling
```

Photometric augmentations:

```
brightness shift
contrast change
color jitter
gaussian noise
```

These augmentations simulate real-world dermoscopic imaging conditions.

---

# 11. Evaluation Metrics

Segmentation performance is evaluated using:

### Dice Coefficient

```
Dice = 2TP / (2TP + FP + FN)
```

### Intersection-over-Union (IoU)

```
IoU = TP / (TP + FP + FN)
```

### Pixel Accuracy

```
accuracy = correct_pixels / total_pixels
```

Dice and IoU are the primary metrics for medical segmentation.

---

# 12. Inference Pipeline

During inference the segmentation pipeline operates as follows:

```
Input Image
      ↓
YOLOv8 Lesion Detection
      ↓
Bounding Box
      ↓
MedSAM Segmentation
      ↓
Lesion Mask
```

The predicted lesion mask is then used for further analysis such as dermoscopic attribute segmentation.

---

# 13. Advantages of Using MedSAM

Compared with training traditional models (e.g., U-Net), MedSAM provides several benefits:

1. **Foundation model initialization**

The model starts from a large-scale pretrained representation.

2. **Better generalization**

SAM was trained on millions of images.

3. **Prompt-based segmentation**

Bounding boxes allow flexible interaction and integration with detection models.

4. **Reduced training data requirement**

Fine-tuning requires significantly less labeled medical data.

---

# 14. Summary

MedSAM training adapts the Segment Anything architecture to medical image segmentation.

Key aspects include:

* bounding box prompt-based training
* fine-tuning SAM image encoder and mask decoder
* BCE + Dice loss for segmentation optimization
* strong preprocessing and augmentation

This approach enables accurate lesion segmentation even with limited annotated medical datasets.
