"""
Model definitions:
  - MedSAM wrapper for lesion segmentation (Stage 2)
  - Attribute segmentation model using SAM encoder + custom decoder (Stage 3)
"""

import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from medsam.config import SAM_MODEL_TYPE, NUM_ATTRIBUTES


# ── MedSAM wrapper ─────────────────────────────────────────────────────
class MedSAM(nn.Module):
    """MedSAM for lesion segmentation.
    Fine-tunes image encoder + mask decoder; prompt encoder is frozen.
    """

    def __init__(self, checkpoint: str = None):
        super().__init__()
        self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint)
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, bbox):
        """
        image: (B, 3, 1024, 1024) normalised
        bbox:  (B, 4) in pixel coords [x_min, y_min, x_max, y_max]
        returns: predicted masks (B, 1, 256, 256)
        """
        image_embedding = self.sam.image_encoder(image)

        results = []
        for i in range(image.shape[0]):
            box_torch = bbox[i:i+1, :].to(image.device)

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embedding[i:i+1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            results.append(low_res_masks)

        return torch.cat(results, dim=0)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── Attribute Segmentation Decoder ─────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttributeDecoder(nn.Module):
    """Lightweight decoder: upsamples SAM encoder features → 5-channel mask."""

    def __init__(self, encoder_dim=256, num_classes=NUM_ATTRIBUTES):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(encoder_dim, 128, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 32)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(16, 16)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        return self.head(x)


class AttributeSegModel(nn.Module):
    """SAM image encoder (frozen) + trainable attribute decoder."""

    def __init__(self, sam_checkpoint: str = None, freeze_encoder: bool = True):
        super().__init__()
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder
        self.decoder = AttributeDecoder(encoder_dim=256, num_classes=NUM_ATTRIBUTES)

        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, image):
        """
        image: (B, 3, 1024, 1024)
        returns: (B, 5, H_out, W_out) logits
        """
        with torch.set_grad_enabled(not self._encoder_frozen()):
            features = self.image_encoder(image)
        logits = self.decoder(features)
        return logits

    def _encoder_frozen(self):
        return not any(p.requires_grad for p in self.image_encoder.parameters())

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
