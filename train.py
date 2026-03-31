#!/usr/bin/env python3
"""
train.py
────────
One-shot training script.  Run this once to train the model:

    python train.py

The best checkpoint is saved to  models/best_model.pth
Training history is saved to     logs/training_history.json
Test evaluation is saved to      logs/evaluation_metrics.json
"""

import os
import random
import argparse

import numpy as np
import torch

import config
from src.model   import build_model
from src.dataset import get_dataloader
from src.trainer import Trainer
from src.evaluate import evaluate


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Bone Fracture Classifier")
    parser.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LR)
    parser.add_argument("--no-oversample", action="store_true",
                        help="Disable weighted sampler for class balancing")
    parser.add_argument("--no-eval",     action="store_true",
                        help="Skip test-set evaluation after training")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = get_dataloader(
        "train",
        batch_size=args.batch_size,
        oversample=not args.no_oversample,
    )
    val_loader   = get_dataloader("val",  batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Train batches : {len(train_loader)}")
    print(f"[INFO] Val   batches : {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(device=device, freeze_backbone=True)
    params = model.count_parameters()
    print(f"[INFO] Parameters → total={params['total']:,}  "
          f"trainable={params['trainable']:,}  frozen={params['frozen']:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(model, device)
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
    )

    print(f"\n[INFO] Best model saved to : {config.MODEL_PATH}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if not args.no_eval:
        print("\n[INFO] Running evaluation on test set …")
        evaluate(model, device)


if __name__ == "__main__":
    main()
