"""
Retrains (fine-tunes) your best model on new misclassified images
stored in data/retrain/misclassified/.
Usage:
    python src/retrain_model.py
"""

import os, torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
MODEL_PATH = "models/best_model.pt"
SAVE_PATH = "models/fine_tuned_model.pt"
MISCLASSIFIED_DIR = "data/retrain/misclassified"
DATA_ROOT = "data/raw"
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4

# ------------------------------------------------
# PREPROCESSING
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# load existing class names
classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])

# if there are retraining samples, fine-tune; otherwise, exit gracefully
if not os.path.exists(MISCLASSIFIED_DIR) or len(os.listdir(MISCLASSIFIED_DIR)) == 0:
    print("❌ No new samples found in data/retrain/misclassified/. Nothing to retrain.")
    exit()

# create labeled dataset automatically (folders grouped by class name)
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------
# TRAIN LOOP
# ------------------------------------------------
print(f"🚀 Fine-tuning on {len(dataset)} total images for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    running_loss = 0
    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {running_loss/len(loader):.4f}")

# ------------------------------------------------
# SAVE UPDATED MODEL
# ------------------------------------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Fine-tuned model saved to {SAVE_PATH}")
