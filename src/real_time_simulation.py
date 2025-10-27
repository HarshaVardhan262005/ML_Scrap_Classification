"""
Real-time (simulated) conveyor loop.

Features:
 - Reads images from a folder (frames are processed at regular intervals).
 - For each frame: classifies using the TorchScript model, logs results to CSV.
 - Prints low-confidence warnings and marks them in CSV.
 - Optional interactive/manual override: press keys to mark prediction correct/incorrect and optionally move misclassified image to retrain queue.

Usage examples:
 - Non-interactive, using images in `data/simulation_frames`:
     python src/real_time_simulation.py --source data/simulation_frames --interval 0.5

 - Use test images automatically (shuffles test set):
     python src/real_time_simulation.py --use-test

 - Interactive manual override enabled (shows images, press keys):
     python src/real_time_simulation.py --source data/simulation_frames --manual

Key controls (when --manual is used and cv2 window visible):
 - 'y' => confirm predicted label (accept)
 - 'n' => mark as misclassified (moves image to data/retrain/misclassified/)
 - 'q' => quit simulation early
"""

import argparse
import os
import time
import csv
import random
from datetime import datetime
import shutil

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

# --------------------------
# Helpers
# --------------------------
def load_class_names(data_root="data/raw"):
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    return classes

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_image_pil(img_pil, model, device, transform):
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    return top_idx, top_prob, probs

# --------------------------
# Main simulation loop
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/simulation_frames", help="Folder containing frames (images) for simulation")
    parser.add_argument("--use-test", action="store_true", help="If set, sample images from data/raw's test split (random)")
    parser.add_argument("--model", default="models/model_torchscript.pt", help="Path to TorchScript model")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between frames")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for low-confidence flag")
    parser.add_argument("--manual", action="store_true", help="Enable manual override UI (keypresses). Requires display.")
    parser.add_argument("--results-csv", default="results/simulation_results.csv", help="CSV file to append results")
    parser.add_argument("--retrain-dir", default="data/retrain/misclassified", help="Where to move misclassified images when manual override marks them")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    os.makedirs(args.retrain_dir, exist_ok=True)

    # device + model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    class_names = load_class_names("data/raw")
    transform = get_transform()

    # Prepare list of images
    if args.use_test:
        # sample images from each class folder (test split is not stored as separate files here),
        # so we randomly sample images from raw/ to simulate frames
        all_images = []
        for cls in class_names:
            cls_folder = os.path.join("data/raw", cls)
            for fn in os.listdir(cls_folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_images.append((os.path.join(cls_folder, fn), cls))
        random.shuffle(all_images)
        frame_list = all_images
    else:
        # take images in provided folder
        if not os.path.isdir(args.source):
            raise FileNotFoundError(f"Source folder '{args.source}' not found. Create it or use --use-test.")
        files = [f for f in os.listdir(args.source) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        frame_list = [(os.path.join(args.source, f), None) for f in files]  # None => unknown true label

    # open (or create) CSV and write header if empty
    csv_exists = os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["timestamp", "frame_path", "true_label", "predicted_label", "confidence", "low_conf_flag", "manual_flag"])

    print("Starting simulation. Press Ctrl+C to stop.")
    idx = 0
    try:
        while idx < len(frame_list):
            frame_path, true_label = frame_list[idx]
            # load image
            pil_img = Image.open(frame_path).convert("RGB")
            top_idx, top_prob, probs = predict_image_pil(pil_img, model, device, transform)
            pred_label = class_names[top_idx]
            low_flag = top_prob < args.threshold

            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"[{timestamp}] Frame: {os.path.basename(frame_path)}  Pred: {pred_label}  Conf: {top_prob:.3f}  LowConf: {low_flag}")

            # log to csv
            with open(args.results_csv, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, frame_path, true_label or "", pred_label, f"{top_prob:.4f}", int(low_flag), 0])

            # Manual override UI (optional)
            if args.manual:
                # show image with prediction text using OpenCV
                # prepare cv2 image
                cv_img = cv2.cvtColor(np.array(pil_img.resize((640, 480))), cv2.COLOR_RGB2BGR)
                cv2.putText(cv_img, f"Pred: {pred_label} ({top_prob:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(cv_img, "Press: [y]=accept [n]=mark_misclassified [q]=quit", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Simulation", cv_img)
                key = cv2.waitKey(0) & 0xFF  # wait until key press
                if key == ord('q'):
                    print("Quit requested by user.")
                    break
                elif key == ord('y'):
                    # accepted, mark manual_flag=1 in CSV
                    with open(args.results_csv, "a", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([timestamp, frame_path, true_label or "", pred_label, f"{top_prob:.4f}", int(low_flag), 1])
                    print("Marked accepted.")
                elif key == ord('n'):
                    # mark as misclassified: move file to retrain dir
                    dest = os.path.join(args.retrain_dir, os.path.basename(frame_path))
                    shutil.copy(frame_path, dest)
                    with open(args.results_csv, "a", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([timestamp, frame_path, true_label or "", pred_label, f"{top_prob:.4f}", int(low_flag), -1])
                    print(f"Marked misclassified and copied to {dest}")
                cv2.destroyAllWindows()
            else:
                # non-interactive: sleep for interval
                time.sleep(args.interval)

            idx += 1

    except KeyboardInterrupt:
        print("Simulation interrupted by user. Exiting...")

    print("Simulation finished.")
