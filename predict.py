#!/usr/bin/env python3
"""
predict.py
──────────
Single-image or batch prediction from the command line.

Usage:
    # Single image
    python predict.py --image path/to/xray.jpg

    # Directory of images
    python predict.py --dir path/to/images/

    # With Grad-CAM heatmap saved alongside each image
    python predict.py --image path/to/xray.jpg --gradcam
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import config
from src.model    import load_model
from src.dataset  import get_transforms
from src.evaluate import GradCAM


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> torch.Tensor:
    transform = get_transforms("val")
    img       = Image.open(image_path).convert("RGB")
    tensor    = transform(img).unsqueeze(0)   # (1, C, H, W)
    return tensor


# ─── Predict ──────────────────────────────────────────────────────────────────

def predict_single(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    gradcam: bool = False,
) -> dict:
    tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    pred_idx    = int(np.argmax(probs))
    pred_label  = config.CLASS_NAMES[pred_idx]
    confidence  = float(probs[pred_idx])

    result = {
        "file":       os.path.basename(image_path),
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(config.CLASS_NAMES, probs)
        },
    }

    # ── Optional Grad-CAM ────────────────────────────────────────────────────
    if gradcam:
        try:
            import cv2
        except ImportError:
            print("[WARNING] opencv-python not installed – skipping Grad-CAM.")
        else:
            cam_gen  = GradCAM(model)
            t_grad   = preprocess(image_path).to(device)
            t_grad.requires_grad_(True)
            heatmap  = cam_gen.generate(t_grad, class_idx=pred_idx)

            orig = np.array(Image.open(image_path).convert("RGB").resize(
                (config.IMG_SIZE, config.IMG_SIZE)
            ))
            heatmap_resized = cv2.resize(heatmap, (config.IMG_SIZE, config.IMG_SIZE))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(orig[:, :, ::-1], 0.6, heatmap_colored, 0.4, 0)

            cam_path = os.path.splitext(image_path)[0] + "_gradcam.jpg"
            cv2.imwrite(cam_path, overlay)
            result["gradcam_path"] = cam_path
            print(f"[INFO] Grad-CAM saved to {cam_path}")

    return result


def predict_dir(
    dir_path: str,
    model: torch.nn.Module,
    device: torch.device,
    gradcam: bool = False,
) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        print(f"[WARNING] No image files found in {dir_path}")
        return []

    results = []
    for fpath in files:
        res = predict_single(fpath, model, device, gradcam=gradcam)
        results.append(res)
        emoji = "🦴" if res["prediction"] == "Fractured" else "✅"
        print(f"{emoji}  {res['file']}  →  {res['prediction']}  ({res['confidence']}%)")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bone Fracture Classifier – Inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single X-ray image")
    group.add_argument("--dir",   type=str, help="Path to a directory of X-ray images")
    parser.add_argument("--gradcam", action="store_true",
                        help="Save Grad-CAM overlay (requires opencv-python)")
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(path=args.model_path, device=device)

    if args.image:
        if not os.path.isfile(args.image):
            print(f"[ERROR] File not found: {args.image}")
            sys.exit(1)
        result = predict_single(args.image, model, device, gradcam=args.gradcam)
        print(f"\nPrediction  : {result['prediction']}")
        print(f"Confidence  : {result['confidence']}%")
        print("Probabilities:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob / 5)
            print(f"  {cls:15s} {prob:5.1f}%  {bar}")
    else:
        predict_dir(args.dir, model, device, gradcam=args.gradcam)


if __name__ == "__main__":
    main()
