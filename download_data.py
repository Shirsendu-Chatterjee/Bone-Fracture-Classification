#!/usr/bin/env python3
"""
download_data.py
────────────────
Downloads the Kaggle bone-fracture dataset and reorganises it into:
  data/
    train/Fractured/   train/Not Fractured/
    val/Fractured/     val/Not Fractured/
    test/Fractured/    test/Not Fractured/

Prerequisites
─────────────
1. Install kaggle API:  pip install kaggle
2. Place your kaggle.json in ~/.kaggle/   (or set KAGGLE_USERNAME + KAGGLE_KEY env vars)
   Get it from: https://www.kaggle.com/account → "Create New API Token"
"""

import os
import sys
import shutil
import zipfile

import config

# ─── Helpers ──────────────────────────────────────────────────────────────────

def check_kaggle_auth():
    """Verify the Kaggle credentials exist before attempting download."""
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    user  = os.environ.get("KAGGLE_USERNAME")
    token = os.environ.get("KAGGLE_KEY")
    if not os.path.exists(cred_path) and not (user and token):
        print(
            "\n[ERROR] Kaggle credentials not found.\n"
            "Options:\n"
            "  A) Place kaggle.json in ~/.kaggle/\n"
            "  B) Export KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
            "Download token from: https://www.kaggle.com/account\n"
        )
        sys.exit(1)


def download_dataset():
    """Pull the dataset zip via the Kaggle CLI."""
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()
    api.authenticate()

    zip_dir = os.path.join(config.BASE_DIR, "data_raw")
    os.makedirs(zip_dir, exist_ok=True)

    print(f"[INFO] Downloading dataset: {config.KAGGLE_DATASET}")
    api.dataset_download_files(config.KAGGLE_DATASET, path=zip_dir, unzip=False)

    # Find the downloaded zip
    zips = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
    if not zips:
        print("[ERROR] No zip file found after download.")
        sys.exit(1)

    zip_path = os.path.join(zip_dir, zips[0])
    print(f"[INFO] Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_dir)

    return zip_dir


def _find_split_root(base: str, split: str) -> str | None:
    """
    Walk `base` looking for a directory whose name matches `split`
    (case-insensitive) that contains class sub-folders.
    """
    for root, dirs, _ in os.walk(base):
        if os.path.basename(root).lower() == split.lower():
            # Confirm it actually has sub-folders with images
            for d in dirs:
                sub = os.path.join(root, d)
                imgs = [f for f in os.listdir(sub)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                if imgs:
                    return root
    return None


def organise(raw_dir: str):
    """Copy extracted files into the canonical data/ layout."""
    splits = ["train", "val", "test"]

    for split in splits:
        split_root = _find_split_root(raw_dir, split)
        if split_root is None:
            print(f"[WARNING] Could not locate '{split}' folder – skipping.")
            continue

        dest_split = os.path.join(config.DATA_DIR, split)
        if os.path.exists(dest_split):
            print(f"[INFO] {dest_split} already exists – skipping copy.")
            continue

        print(f"[INFO] Copying {split_root}  →  {dest_split}")
        shutil.copytree(split_root, dest_split)

    # Quick sanity check
    for split in splits:
        split_path = os.path.join(config.DATA_DIR, split)
        if not os.path.exists(split_path):
            continue
        classes = os.listdir(split_path)
        counts = {c: len(os.listdir(os.path.join(split_path, c))) for c in classes}
        print(f"  [{split}] {counts}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if os.path.exists(config.TRAIN_DIR):
        print("[INFO] Data directory already exists. Skipping download.")
        return

    check_kaggle_auth()
    raw_dir = download_dataset()
    organise(raw_dir)
    print("\n[DONE] Dataset ready at:", config.DATA_DIR)


if __name__ == "__main__":
    main()
