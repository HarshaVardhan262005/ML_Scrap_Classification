♻️ ML Scrap Classification – Performance Report

📘 Overview

This project demonstrates an end-to-end machine learning pipeline for automated waste classification using computer vision and deep learning (PyTorch).
It simulates a real-world conveyor belt system that classifies recyclable materials in real time, logs predictions, and saves uncertain samples for retraining.


🧠 Model & Dataset Summary

| Component        | Details                                                   |
| ---------------- | --------------------------------------------------------- |
| **Dataset**      | [TrashNet Dataset](https://github.com/garythung/trashnet) |
| **Classes**      | 6 — cardboard, glass, metal, paper, plastic, trash        |
| **Total Images** | ~2500 labeled images                                      |
| **Split**        | Train 70% • Validation 15% • Test 15%                     |
| **Input Size**   | 224 × 224 × 3                                             |
| **Framework**    | PyTorch (CPU)                                             |
| **Base Model**   | ResNet-18 (Pretrained on ImageNet)                        |
| **Approach**     | Transfer Learning (only final layer retrained)            |


📊 Performance Metrics

| Metric                | Score   |
| --------------------- | ------- |
| **Accuracy**          | **92%** |
| **Precision**         | 0.92    |
| **Recall**            | 0.91    |
| **F1-Score**          | 0.92    |
| **Classes Evaluated** | 6       |
| **Test Samples**      | 380     |



📊 Classification Report Summary:

| Class     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| Cardboard | 1.00      | 0.95   | 0.97     |
| Glass     | 0.92      | 0.91   | 0.92     |
| Metal     | 0.90      | 0.93   | 0.92     |
| Paper     | 0.94      | 0.96   | 0.95     |
| Plastic   | 0.87      | 0.89   | 0.88     |
| Trash     | 0.91      | 0.83   | 0.87     |


Example Simulation Output:
🚀 Starting conveyor simulation...
📸 plastic1.jpg -> plastic (0.987)
📸 plastic2.jpg -> glass (0.61) ⚠️ Low confidence
✅ Simulation complete. Results saved to results/conveyor_results.csv
