# Training Pipeline for ISIC 2018 Task 1 and Task 2

**Architecture: YOLOv8 + MedSAM**

## 1. Overview

This project focuses on two tasks from the ISIC 2018 Skin Lesion Analysis Challenge:

* **Task 1 — Lesion Segmentation**
* **Task 2 — Dermoscopic Attribute Segmentation**

Instead of training classical segmentation models from scratch (e.g., U-Net), this pipeline leverages **modern foundation models and detection frameworks**:

* **YOLOv8** for lesion localization and coarse segmentation
* **MedSAM** for precise lesion boundary segmentation
* **MedSAM-based multi-label segmentation head** for dermoscopic attribute detection

This architecture improves robustness against:

* extreme scale variation
* small dermoscopic structures
* annotation noise
* limited dataset size

---

# 2. Dataset

The ISIC 2018 dataset contains dermoscopic images annotated for three tasks.
This pipeline uses the annotations for:

* **Task 1:** lesion boundary segmentation
* **Task 2:** dermoscopic attribute masks

Dataset structure:

```
dataset/
    images/
        ISIC_0000001.jpg
        ISIC_0000002.jpg

    lesion_masks/
        ISIC_0000001_segmentation.png

    attribute_masks/
        ISIC_0000001_pigment_network.png
        ISIC_0000001_globules.png
        ISIC_0000001_streaks.png
        ISIC_0000001_milia_like_cyst.png
        ISIC_0000001_negative_network.png
```

The attribute masks correspond to five dermoscopic structures:

* pigment network
* globules
* streaks
* milia-like cysts
* negative network

---

# 3. Preprocessing Pipeline

Dermoscopic images vary significantly in resolution and scale.

Observed dataset statistics:

* resolution range: **576×542 → 6708×4461**
* lesion size range: **0.4% → 98.7% of image**

Therefore standardized preprocessing is required.

## 3.1 Image Normalization

Images are resized so that the longest side equals:

```
longest_side = 1024
```

Aspect ratio is preserved.

---

## 3.2 Random Cropping

During training random crops are extracted:

```
crop_size = 512 × 512
```

This reduces GPU memory usage while increasing training diversity.

---

## 3.3 Data Augmentation

Augmentation improves robustness to clinical imaging artifacts.

Applied augmentations:

Geometric:

* random rotation
* horizontal flip
* vertical flip
* random scaling

Photometric:

* color jitter
* brightness/contrast shifts
* gaussian noise

Clinical artifacts:

* synthetic hair occlusion
* gaussian blur
* illumination shifts

---

# 4. Model Architecture

The system follows a **three-stage architecture**:

```
Input Dermoscopic Image
        │
        ▼
YOLOv8 Lesion Detection
        │
        ▼
MedSAM Lesion Segmentation
        │
        ▼
MedSAM Attribute Segmentation
```

This modular design allows each model to specialize in a specific task.

---

# 5. Stage 1 — Lesion Detection (YOLOv8)

## Objective

Detect the bounding box of the lesion within the dermoscopic image.

```
Input: dermoscopic image
Output: lesion bounding box
```

---

## Model

YOLOv8 detection model is used.

Architecture:

```
Backbone: CSPDarknet
Neck: PAN-FPN
Head: Detection head
```

YOLOv8 provides:

* fast inference
* strong performance for object localization
* robustness to scale variation

---

## Training Target

The model predicts:

```
[x_center, y_center, width, height]
```

for the lesion region.

---

## Loss Functions

YOLOv8 training uses:

```
Box loss
Objectness loss
Classification loss
```

However only **one class (lesion)** is used.

---

# 6. Stage 2 — Lesion Segmentation (MedSAM)

## Objective

Generate an accurate lesion boundary mask.

```
Input: dermoscopic image + lesion bounding box
Output: binary lesion mask
```

---

## MedSAM Model

MedSAM is a medical adaptation of the Segment Anything Model.

Architecture components:

```
Image Encoder: Vision Transformer
Prompt Encoder: bounding box prompt
Mask Decoder: segmentation mask generator
```

The lesion bounding box predicted by YOLOv8 is used as the **SAM prompt**.

Pipeline:

```
image
  + bbox prompt
        │
        ▼
MedSAM
        │
        ▼
lesion segmentation mask
```

---

## Training Strategy

The MedSAM encoder is typically **frozen**.

Only the segmentation decoder is fine-tuned on ISIC masks.

Loss function:

```
Dice Loss
+
Binary Cross Entropy
```

Evaluation metrics:

* Dice coefficient
* Intersection-over-Union (IoU)

---

# 7. Stage 3 — Dermoscopic Attribute Segmentation (Task 2)

## Objective

Detect dermoscopic structures inside the lesion.

Output masks:

```
pigment network
globules
streaks
milia-like cyst
negative network
```

---

## Input Preparation

The lesion mask from Task 1 is used to crop the lesion region.

Pipeline:

```
original image
        │
        ▼
lesion mask
        │
        ▼
lesion crop
        │
        ▼
attribute segmentation
```

This focuses the model on clinically relevant regions.

---

## Model Architecture

Attribute segmentation uses **MedSAM encoder + multi-label segmentation head**.

Architecture:

```
MedSAM Image Encoder
        │
        ▼
Feature Map
        │
        ▼
Multi-label Segmentation Decoder
        │
        ▼
5-channel output masks
```

Output tensor:

```
[5, H, W]
```

Each channel corresponds to one dermoscopic attribute.

---

# 8. Handling Dataset Challenges

The ISIC dataset presents several challenges.

## 8.1 Severe Class Imbalance

Example imbalance:

```
pigment network = 59%
streaks = 3.5%
```

Solutions:

* weighted BCE loss
* focal loss
* oversampling rare classes

---

## 8.2 Small Structures

Dermoscopic attributes are extremely small:

```
pixel coverage: 0.06% → 3.7%
```

Solutions:

* high resolution training
* feature pyramid learning
* multi-scale augmentation

---

## 8.3 Missing Annotations

Approximately **21% of images have no attribute labels**.

Solutions:

* ignore empty-mask loss
* semi-supervised learning
* pseudo-label generation

---

# 9. Training Procedure

Training is performed in three sequential stages.

## Stage 1

Train YOLOv8 lesion detector.

```
input: dermoscopic images
output: lesion bounding boxes
```

---

## Stage 2

Train MedSAM lesion segmentation model.

```
input: image + YOLO bbox
output: lesion mask
```

---

## Stage 3

Train dermoscopic attribute segmentation model.

```
input: cropped lesion region
output: 5 attribute masks
```

---

# 10. Inference Pipeline

During inference the system runs sequentially.

```
Dermoscopic Image
        │
        ▼
YOLOv8 Detection
        │
        ▼
MedSAM Lesion Segmentation
        │
        ▼
Lesion Crop
        │
        ▼
Attribute Segmentation
```

Outputs:

* lesion boundary mask
* dermoscopic attribute maps

These outputs can be used to support dermatological diagnosis.

---

# 11. Advantages of the Proposed Pipeline

The YOLOv8 + MedSAM architecture provides several advantages:

1. **Improved localization**

YOLOv8 accurately identifies lesion regions.

2. **High-quality segmentation**

MedSAM produces precise boundaries due to large-scale pretraining.

3. **Better detection of small structures**

Cropping lesion regions increases effective resolution.

4. **Robustness to annotation noise**

Foundation models reduce overfitting to small datasets.

---

# 12. Summary

This pipeline combines modern detection and foundation segmentation models to analyze dermoscopic images.

Key components:

* YOLOv8 lesion localization
* MedSAM lesion segmentation
* MedSAM-based dermoscopic attribute segmentation

The system addresses major dataset challenges including:

* class imbalance
* small dermoscopic structures
* annotation noise
* large resolution variation

This architecture provides a strong foundation for explainable AI systems for skin lesion analysis.
