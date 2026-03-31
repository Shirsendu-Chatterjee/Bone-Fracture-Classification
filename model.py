"""
src/model.py
────────────
EfficientNetB0 transfer-learning model for binary X-ray classification.
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Architecture ─────────────────────────────────────────────────────────────

class BoneFractureClassifier(nn.Module):
    """
    EfficientNetB0 backbone with a custom classification head.

    Architecture
    ────────────
    EfficientNetB0 (pretrained on ImageNet, backbone frozen initially)
      └─ AdaptiveAvgPool2d  →  1×1
      └─ Dropout(p=DROPOUT)
      └─ Linear(1280, 256)
      └─ ReLU
      └─ Dropout(p=DROPOUT/2)
      └─ Linear(256, NUM_CLASSES)
    """

    def __init__(
        self,
        num_classes: int  = config.NUM_CLASSES,
        pretrained:  bool = config.PRETRAINED,
        dropout:     float= config.DROPOUT,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────────────
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove the original classifier
        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        # ── Custom head ──────────────────────────────────────────────────────
        in_features = backbone.classifier[1].in_features   # 1280 for B0
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # ── Convenience ──────────────────────────────────────────────────────────

    def unfreeze_backbone(self, from_layer: int = -3):
        """
        Unfreeze the last `abs(from_layer)` feature blocks for fine-tuning.
        Call this after initial training converges.
        """
        layers = list(self.features.children())
        for layer in layers[from_layer:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"[INFO] Unfrozen backbone layers from index {len(layers) + from_layer} onward.")

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int  = config.NUM_CLASSES,
    pretrained:  bool = config.PRETRAINED,
    freeze_backbone: bool = True,
    device: torch.device | None = None,
) -> BoneFractureClassifier:
    """Build and move the model to `device`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoneFractureClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    return model.to(device)


def load_model(
    path:  str          = config.MODEL_PATH,
    device: torch.device | None = None,
) -> BoneFractureClassifier:
    """Load a saved model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Model file not found at '{path}'.\n"
            "Run   python train.py   to train and save the model first."
        )

    checkpoint = torch.load(path, map_location=device)

    model = BoneFractureClassifier(
        num_classes=checkpoint.get("num_classes", config.NUM_CLASSES),
    )

    # Support both raw state-dicts and wrapped checkpoints
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", "N/A")
    print(f"[INFO] Loaded model from '{path}'  (val_acc={val_acc})")
    return model
