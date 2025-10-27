"""
Simple inference script for one image using the TorchScript model.

Usage:
    python src/inference_script.py --image path/to/img.jpg --model models/model_torchscript.pt --threshold 0.6

Outputs:
    Prints predicted class and confidence.
"""
import argparse
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np

# --------------------------
# helper: load class names
# --------------------------
def load_class_names(data_root="data/raw"):
    # ImageFolder order is alphabetical by folder name -> classes
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    return classes

# --------------------------
# transforms (should match training)
# --------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# --------------------------
# predict function
# --------------------------
def predict(image_path, model, device, class_names, transform, topk=1):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # shape: 1,3,224,224
    with torch.no_grad():
        out = model(x)  # TorchScript returns logits
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    return class_names[top_idx], top_prob, probs

# --------------------------
# main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/model_torchscript.pt", help="Path to TorchScript model")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for low-confidence flag")
    parser.add_argument("--data-root", default="data/raw", help="Root folder where class subfolders are located")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    class_names = load_class_names(args.data_root)
    transform = get_transform()

    predicted_class, confidence, probs = predict(args.image, model, device, class_names, transform)
    low_flag = confidence < args.threshold

    print(f"Image: {args.image}")
    print(f"Predicted: {predicted_class} \t Confidence: {confidence:.4f}")
    print("Low confidence flag:" , low_flag)

if __name__ == "__main__":
    main()
