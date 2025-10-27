"""
Simulates a conveyor belt capturing frames at intervals
and classifying each frame in real time.

Usage:
    python src/conveyor_simulation.py --folder data/raw/plastic/plastic --interval 1.0
"""

import os
import time
import csv
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
MODEL_PATH = "models/model_torchscript.pt"
DATA_ROOT = "data/raw"
RESULTS_CSV = "results/conveyor_results.csv"
THRESHOLD = 0.85

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def load_class_names(data_root=DATA_ROOT):
    """Load class names from dataset structure."""
    return sorted([d for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d))])

def get_transform():
    """Image preprocessing identical to training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def find_true_label(path):
    """Extract true class from path name (works with nested folders)."""
    parts = path.replace("\\", "/").split("/")
    for p in reversed(parts):
        if p.lower() in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
            return p.lower()
    return "unknown"

def predict(image_path, model, device, transform, class_names):
    """Predict single image."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return class_names[idx], conf

# ------------------------------------------------
# MAIN SIMULATION
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Conveyor Belt Simulation")
    parser.add_argument("--folder", required=True, help="Folder containing test frames (images)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Low-confidence threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(DATA_ROOT)
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    transform = get_transform()

    # ensure results folder exists
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    # prepare CSV file
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Predicted", "Confidence", "LowConfidence", "TrueLabel", "Timestamp"])

        # loop through images like a live feed
        images = [os.path.join(args.folder, i) for i in os.listdir(args.folder)
                  if i.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()

        print(f"🚀 Starting conveyor simulation with {len(images)} frames...\n")

        for img_path in images:
            pred, conf = predict(img_path, model, device, transform, class_names)
            low_flag = conf < args.threshold
            true_label = find_true_label(img_path)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # print simulated frame classification
            print(f"📸 {os.path.basename(img_path)} -> {pred} ({conf:.3f}) {'⚠️' if low_flag else ''}")

            # log to CSV
            writer.writerow([os.path.basename(img_path), pred, round(conf, 4), low_flag, true_label, timestamp])
            f.flush()  # save after each frame

            time.sleep(args.interval)  # simulate frame interval

    print(f"\n✅ Simulation complete. Results saved to: {RESULTS_CSV}")

# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    main()
