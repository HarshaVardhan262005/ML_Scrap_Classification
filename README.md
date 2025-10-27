♻️ ML Scrap Classification Project
🧩 Overview

This project automatically classifies waste items (scrap materials) into six categories using deep learning and computer vision.
It mimics an industrial conveyor belt that classifies incoming items in real-time, logging results and confidence levels.

🧠 Dataset Used

Dataset: TrashNet Dataset
📘 Description

The TrashNet dataset contains ~2500 labeled images divided into 6 categories:

1.cardboard
2.glass
3.metal
4.paper
5.plastic
6.trash

Each image shows a single object on a neutral background (e.g., bottles, cans, cardboard boxes, etc.).

🎯 Why This Dataset?

It represents real-world recyclable materials.

Ideal for simulating an automated waste segregation system.

Publicly available, balanced, and widely used in sustainability-related ML projects.

🧱 Architecture & Training Process

🧠 Model Architecture

Base Model: ResNet-18 (pretrained on ImageNet)

Approach: Transfer Learning
Only the final fully connected (FC) layer is retrained to classify 6 categories.

Framework: PyTorch (CPU build)

Input Size: 224×224×3

⚙️ Training Pipeline

Data Augmentation:

Random horizontal flips, rotations, normalization

Ensures model generalization on unseen lighting/backgrounds.

Data Split:

Train: 70%

Validation: 15%

Test: 15%

Optimizer: Adam (lr = 1e-4)

Loss Function: CrossEntropyLoss

Early Stopping: Stops training when validation accuracy plateaus.

Checkpoints: Saves the best model (best_model.pt).

Output Formats:

Regular PyTorch model (best_model.pt)

TorchScript model (model_torchscript.pt) for lightweight deployment.


🧮 Model Summary

| Layer Type           | Details                                     |
| -------------------- | ------------------------------------------- |
| Convolutional Layers | Extracts low- and mid-level visual features |
| Residual Blocks      | Improves gradient flow, faster convergence  |
| Fully Connected (FC) | 512 → 6 neurons (one per class)             |
| Activation           | ReLU                                        |
| Output               | Softmax probabilities for 6 categories      |

🧩 Deployment Decisions

| Component                   | Decision                                                                           | Reason                                    |
| --------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| **Format**                  | TorchScript (`.pt`)                                                                | Portable, optimized for CPU environments  |
| **Inference Engine**        | PyTorch runtime                                                                    | Simple, compatible with VS Code           |
| **Simulation**              | Python loop mimicking video frame capture                                          | Avoids hardware/video camera dependencies |
| **Confidence Thresholding** | 0.85                                                                               | Reduces confidently wrong predictions     |
| **Active Learning**         | Misclassified or low-confidence samples auto-saved to `data/retrain/misclassified` | Continuous improvement of the model       |


🗂️ Folder Structure

ML_Scrap_Classification/
│
├── data/
│   ├── raw/
│   │   ├── cardboard/cardboard/...
│   │   ├── glass/glass/...
│   │   ├── metal/metal/...
│   │   ├── paper/paper/...
│   │   ├── plastic/plastic/...
│   │   └── trash/trash/...
│   ├── processed/
│   └── retrain/
│       └── misclassified/
│
├── models/
│   ├── best_model.pt
│   ├── model_torchscript.pt
│   └── fine_tuned_model.pt
│
├── results/
│   ├── confusion_matrix.png
│   └── conveyor_results.csv
│
├── src/
│   ├── dataset_preparation.py
│   ├── train_model.py
│   ├── inference_robust.py
│   ├── retrain_model.py
│   └── conveyor_simulation.py
│
└── README.md



⚙️ How to Run
1️⃣ Setup
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib tqdm pillow scikit-learn


2️⃣ Dataset Preparation
Download the TrashNet dataset and place all folders in data/raw/.
Then run:
python src/dataset_preparation.py

Output:
✅ Dataset prepared!
Train: 1768 | Val: 379 | Test: 380

3️⃣ Model Training
python src/train_model.py
After training:
✅ Model saved to models/best_model.pt and model_torchscript.pt
Accuracy: 92%


4️⃣ Inference (Single Image)
python src/inference_robust.py --image data/raw/plastic/plastic/plastic1.jpg --threshold 0.85 --save-uncertain
Sample Output:
📸 Image: data/raw/plastic/plastic/plastic1.jpg
🔍 Predicted: plastic
📊 Confidence: 0.9723
✅ Correct prediction: True
⚠️ Low confidence flag: False


5️⃣ Conveyor Simulation (CSV Logging)
Simulates live image capture at intervals:
python src/conveyor_simulation.py --folder data/raw/plastic/plastic --interval 1.0
Output example:
🚀 Starting conveyor simulation...
📸 plastic1.jpg -> plastic (0.987)
📸 plastic2.jpg -> glass (0.61) ⚠️
✅ Simulation complete. Results saved to results/conveyor_results.csv
| Frame        | Predicted | Confidence | LowConfidence | TrueLabel | Timestamp           |
| ------------ | --------- | ---------- | ------------- | --------- | ------------------- |
| plastic1.jpg | plastic   | 0.987      | False         | plastic   | 2025-10-27 16:21:02 |
| plastic2.jpg | glass     | 0.610      | True          | plastic   | 2025-10-27 16:21:03 |


📊 Performance Summary
| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **92%**   |
| Precision | 0.92      |
| Recall    | 0.91      |
| F1-score  | 0.92      |
| Classes   | 6         |
| Model     | ResNet-18 |



🧩 Key Features

✅ Transfer Learning (ResNet18)
✅ Early Stopping & Checkpoints
✅ TorchScript deployment
✅ Confidence Thresholding
✅ Real-time Conveyor Simulation
✅ Active Learning (Automatic Retraining)
✅ CSV Logging for Results


🧾 Conclusion

This project demonstrates a full end-to-end ML pipeline:

Data processing

Model training

Real-time inference

Active learning & retraining

Deployment-ready TorchScript model

With ~92% accuracy, it’s a strong baseline for AI-driven waste segregation and can be scaled to real conveyor systems using cameras and Raspberry Pi–based controllers.




👨‍💻 Author

Harsha Vardhan K (CSE)
ML Intern Assignment — 2025
Department of Computer Science and Engineering



✅ Deliverables Included

Source Code (src/)

Dataset split and processed data

Trained models (best_model.pt, model_torchscript.pt)

Conveyor simulation + result CSV

Documentation (this README.md)
