"""
src/dataset.py
──────────────
PyTorch Dataset & DataLoader factory for bone fracture X-ray images.
"""

import os
from typing import Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return augmentation pipeline.
    Training set gets heavier augmentation; val/test only normalise.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XRayDataset(datasets.ImageFolder):
    """Thin wrapper around ImageFolder that converts grayscale X-rays to RGB."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")   # X-rays are often single-channel
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_dataloader(
    split: str,
    batch_size: int  = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    shuffle: Optional[bool] = None,
    oversample: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for `split` (train | val | test).

    Parameters
    ----------
    oversample : bool
        If True, use WeightedRandomSampler to balance classes (useful when
        the dataset is imbalanced between Fractured / Not Fractured).
    """
    root = {
        "train": config.TRAIN_DIR,
        "val":   config.VAL_DIR,
        "test":  config.TEST_DIR,
    }[split]

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"[ERROR] {split} directory not found at '{root}'.\n"
            "Run   python download_data.py   first."
        )

    dataset = XRayDataset(root=root, transform=get_transforms(split))

    # Default shuffle behaviour
    if shuffle is None:
        shuffle = (split == "train")

    sampler = None
    if oversample and split == "train":
        class_counts  = torch.bincount(torch.tensor(dataset.targets))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[dataset.targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False   # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def get_class_names(split: str = "train") -> list:
    """Return class names as sorted by ImageFolder (matches config.CLASS_NAMES)."""
    root = {"train": config.TRAIN_DIR, "val": config.VAL_DIR, "test": config.TEST_DIR}[split]
    if not os.path.isdir(root):
        return config.CLASS_NAMES
    dataset = XRayDataset(root=root, transform=get_transforms("val"))
    return dataset.classes
