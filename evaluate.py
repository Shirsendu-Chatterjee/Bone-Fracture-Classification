"""
src/evaluate.py
───────────────
Evaluation utilities: full test-set metrics, confusion matrix, Grad-CAM.
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import get_dataloader, get_class_names


# ─── Full evaluation ──────────────────────────────────────────────────────────

def evaluate(model: torch.nn.Module, device: torch.device) -> dict:
    """
    Run inference on the test split and return a metrics dict containing:
      accuracy, per-class precision/recall/F1, AUC, confusion matrix.
    """
    loader      = get_dataloader("test", shuffle=False)
    class_names = get_class_names("test")

    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds).tolist()

    # AUC (binary)
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 0])
    except ValueError:
        auc = None

    metrics = {
        "accuracy":             round(float(accuracy), 6),
        "auc":                  round(float(auc), 6) if auc else None,
        "classification_report": report,
        "confusion_matrix":     cm,
        "class_names":          class_names,
    }

    # Pretty print
    print(f"\n{'='*55}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    if auc:
        print(f"  AUC           : {auc:.4f}")
    print(f"{'='*55}")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("Confusion Matrix:")
    print(np.array(cm))

    # Save
    out_path = os.path.join(config.LOG_DIR, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to {out_path}")

    return metrics


# ─── GradCAM (lightweight, no extra libs) ─────────────────────────────────────

class GradCAM:
    """
    Minimal Grad-CAM implementation for EfficientNetB0.
    Hooks onto the last convolutional layer (features[-1]).
    """

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target = list(self.model.features.children())[-1]

        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self, image_tensor: torch.Tensor, class_idx: int | None = None):
        """
        Returns a (H, W) numpy heatmap normalised to [0, 1].
        `image_tensor` must be (1, C, H, W).
        """
        self.model.eval()
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
