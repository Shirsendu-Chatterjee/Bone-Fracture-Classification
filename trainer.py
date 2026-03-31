"""
src/trainer.py
──────────────
Full training loop: mixed-precision, early stopping, LR scheduling,
per-epoch metrics logging, and best-model checkpointing.
"""

import os
import sys
import time
import json
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import get_dataloader, get_class_names


# ─── Core loop ────────────────────────────────────────────────────────────────

def run_epoch(
    model:       nn.Module,
    loader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer | None,
    scaler:      GradScaler | None,
    device:      torch.device,
    is_train:    bool,
) -> tuple[float, float]:
    """Single epoch forward (+ backward if `is_train`).  Returns (loss, acc)."""
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train and scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model   = model
        self.device  = device
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_val_acc   = 0.0
        self.best_state     = None
        self.patience_count = 0

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs:   int   = config.NUM_EPOCHS,
        lr:           float = config.LR,
        weight_decay: float = config.WEIGHT_DECAY,
        patience:     int   = config.PATIENCE,
    ):
        """Full training run with early stopping and fine-tuning phase."""
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=config.LR_STEP, gamma=config.LR_GAMMA)
        scaler    = GradScaler() if self.device.type == "cuda" else None

        fine_tune_triggered = False

        print(f"\n{'='*60}")
        print(f"  Training on {self.device}  |  epochs={num_epochs}  |  lr={lr}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # ── Phase switch: unfreeze backbone after patience halfway ─────
            if (not fine_tune_triggered
                    and epoch > num_epochs // 2
                    and self.patience_count >= 2):
                print("\n[INFO] Switching to fine-tuning – unfreezing last 3 backbone blocks …")
                self.model.unfreeze_backbone(from_layer=-3)
                # Rebuild optimiser with lower LR for backbone
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr * 0.1, weight_decay=weight_decay,
                )
                scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
                fine_tune_triggered = True
                self.patience_count = 0

            train_loss, train_acc = run_epoch(
                self.model, train_loader, criterion, optimizer, scaler, self.device, True
            )
            val_loss, val_acc = run_epoch(
                self.model, val_loader, criterion, None, None, self.device, False
            )
            scheduler.step()

            # ── Logging ──────────────────────────────────────────────────
            elapsed = time.time() - t0
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            flag = "  ✓ best" if val_acc > self.best_val_acc else ""
            print(
                f"Epoch [{epoch:02d}/{num_epochs}] "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"({elapsed:.1f}s){flag}"
            )

            # ── Checkpoint best model ─────────────────────────────────────
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state   = copy.deepcopy(self.model.state_dict())
                self.patience_count = 0
                self._save_checkpoint(val_acc)
            else:
                self.patience_count += 1

            # ── Early stopping ────────────────────────────────────────────
            if self.patience_count >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs.")
                break

        # Restore best weights
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        self._save_history()
        print(f"\n[DONE] Best val_acc = {self.best_val_acc:.4f}")
        return self.history

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, val_acc: float):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "val_acc":          round(val_acc, 6),
                "num_classes":      config.NUM_CLASSES,
                "class_names":      config.CLASS_NAMES,
                "img_size":         config.IMG_SIZE,
                "backbone":         config.BACKBONE,
            },
            config.MODEL_PATH,
        )

    def _save_history(self):
        history_path = os.path.join(config.LOG_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[INFO] Training history saved to {history_path}")
