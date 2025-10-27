"""
Robust single-image inference that supports nested folder structures
(e.g., data/raw/plastic/plastic/plastic1.jpg).

Usage:
    python src/inference_robust.py --image data/raw/plastic/plastic/plastic1.jpg --threshold 0.85 --save-uncertain
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import shutil
import time

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
MODEL_PATH = "models/model_torchscript.pt"
DATA_ROOT = "data/raw"
RETRAIN_DIR = "data/retrain/misclassified"
THRESHOLD = 0.8  # default confidence threshold

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def find_class_folder(image_path):
    """
    Automatically find the class folder name from nested paths.
    Example: data/raw/plastic/plastic/plastic1.jpg -> 'plastic'
    """
    parts = image_path.replace("\\", "/").split("/")
    for p in reversed(parts):
        if p.lower() in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
            return p.lower()
    return None

def load_class_names(data_root=DATA_ROOT):
    """Load class names alphabetically from the first folder level."""
    return sorted([d for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d))])

def get_transform():
    """Preprocessing identical to training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict(image_path, model, device, class_names, transform):
    """Run single-image prediction."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    pred_class = class_names[top_idx]
    return pred_class, top_prob, probs

# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Robust inference supporting nested folders")
    parser.add_argument("--image", required=True, help="Path to image for prediction")
    parser.add_argument("--model", default=MODEL_PATH, help="TorchScript model path")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Confidence threshold")
    parser.add_argument("--smooth", type=int, default=1, help="Average multiple predictions for stability")
    parser.add_argument("--save-uncertain", action="store_true", help="Save low-confidence images to retrain folder")
    args = parser.parse_args()

    # Environment setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RETRAIN_DIR, exist_ok=True)

    # Load model and class labels
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    class_names = load_class_names(DATA_ROOT)
    transform = get_transform()

    # Predict multiple times for smoothing
    smoothed_probs = np.zeros(len(class_names))
    for i in range(args.smooth):
        pred_class, conf, probs = predict(args.image, model, device, class_names, transform)
        smoothed_probs += probs
        time.sleep(0.05)
    smoothed_probs /= args.smooth

    top_idx = int(np.argmax(smoothed_probs))
    top_prob = float(smoothed_probs[top_idx])
    predicted_class = class_names[top_idx]
    low_conf_flag = top_prob < args.threshold

    # Attempt to infer true label from path (optional)
    true_label = find_class_folder(args.image)
    correct_flag = (predicted_class == true_label)

    print("\n📸 Image:", args.image)
    print(f"🔍 Predicted: {predicted_class}")
    print(f"📊 Confidence: {top_prob:.4f}")
    print(f"🧾 True label (from folder): {true_label}")
    print(f"✅ Correct prediction: {correct_flag}")
    print(f"⚠️ Low confidence flag: {low_conf_flag}")

    # Save uncertain or incorrect predictions for retraining
    if (low_conf_flag or not correct_flag) and args.save_uncertain:
        dest = os.path.join(RETRAIN_DIR, os.path.basename(args.image))
        shutil.copy(args.image, dest)
        print(f"💾 Saved to retrain folder: {dest}")

    print("\n✅ Inference complete.")

# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    main()
