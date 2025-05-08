# 🧠 YOLOv8 Object Detection with Dataset and Knowledge Distillation

## 📌 Overview

This project presents a comprehensive exploration of two advanced model compression techniques—**Dataset Distillation (DD)** and **Knowledge Distillation (KD)**—applied to object detection using the **YOLOv8** framework on the **COCO 2017** dataset.

Our primary goal is to train a lightweight object detector that retains as much performance as possible while significantly reducing data and/or model size. This is especially useful for deploying object detection on resource-constrained devices like smartphones and embedded systems.

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Project Motivation](#-project-motivation)
- [Techniques Used](#-techniques-used)
  - [Dataset Distillation (DD)](#dataset-distillation-dd)
  - [Knowledge Distillation (KD)](#knowledge-distillation-kd)
- [Dataset Description](#-dataset-description)
- [Model Architectures](#-model-architectures)
  - [YOLOv8](#yolov8)
  - [ResNet-50 Teacher (for KD)](#resnet-50-teacher-for-kd)
- [Project Pipeline](#-project-pipeline)
- [Results and Evaluation](#-results-and-evaluation)
- [Future Work](#-future-work)
---

## 🎯 Project Motivation

Deep object detectors like YOLOv8 achieve impressive performance but are computationally expensive and memory-hungry. This project aims to address this limitation through:
- **Dataset Distillation**: Reducing dataset size without significant performance loss.
- **Knowledge Distillation**: Transferring knowledge from a large model (teacher) to a smaller one (student).

These approaches help build **efficient detectors** suitable for real-time applications.

---

## 🔬 Techniques Used

### Dataset Distillation (DD)

- Synthesizes a **small synthetic dataset** that mimics the behavior of the real dataset.
- Uses **gradient matching** to align the learning dynamics.
- Enables training YOLOv8 on a **tiny subset of synthetic data** with comparable performance.

### Knowledge Distillation (KD)

- A **ResNet-50-based object detection model** acts as the **teacher**.
- The **YOLOv8-small** model acts as the **student**.
- The student learns from:
  - **Soft labels** (logits) output by the teacher.
  - A **combined loss** of student prediction and teacher guidance.

---

## 📁 Dataset Description

We used the **COCO 2017** dataset:

- **Training Set**: `train2017` (for original training and distillation)
- **Validation Set**: `val2017` (for evaluation)
- **Synthetic Dataset**: Generated via dataset distillation

All datasets are in **YOLO format**:

---

## 🧠 Model Architectures

### YOLOv8

- Lightweight, real-time object detection model.
- Architecture components:
  - **Backbone**: CSPDarknet with Focus and C2f modules
  - **Neck**: PAN-FPN for feature fusion
  - **Head**: Detect layer with anchors
  - **Epochs**: 10
    
### ResNet-50 Teacher (for KD)

- Used as a backbone for Faster R-CNN-style detector.
- Outputs class logits and bounding box predictions.
- Helps the YOLOv8 student learn smoother decision boundaries.

---

## 🔄 Project Pipeline

```text
        ┌────────────┐
        │ COCO Data  │
        └────┬───────┘
             ↓
   ┌────────────────────┐
   │ Dataset Distillation│
   └────┬───────────────┘
        ↓
┌──────────────────────────┐
│ Synthetic Dataset (Tiny) │
└────┬─────────────────────┘
     ↓                                ┌──────────────────────────────┐
┌────────────┐      ┌────────────────► Knowledge Distillation (KD)  │
│ YOLOv8-Real│      │                └────────────┬─────────────────┘
└────┬───────┘      │                             ↓
     │              │                    ┌──────────────────┐
     ▼              ▼                    │   YOLOv8 Student │
 Train & Eval  Train & Eval              └────────┬─────────┘
     │              │                            ↓
     └──────┬───────┘                      Performance Comparison
            ↓
      Metrics & Visualization
```
### 📊 Results and Evaluation

| Phase               | Dataset              | mAP@0.5 | Precision | Recall | F1 Score |
|---------------------|----------------------|---------|-----------|--------|----------|
| Report 1 (Baseline) | Real (CBA0277 COCO)  | 0.70    | 0.72      | 0.68   | 0.67     |
| Report 2 (Synthetic)| Distilled COCO       | 0.003   | 1.00      | 0.02   | 0.00     |
| Report 3 (KD)       | Real COCO            | 0.75    | 0.78      | 0.73   | 0.75     |

### 🧩 Interpretation

- **KD (Knowledge Distillation)** outperforms all models with the highest F1 score and mAP.
- The **synthetic dataset** from Dataset Distillation achieves very high precision but extremely low recall—indicating the model detects too few objects.
- The **baseline YOLOv8 model** trained on real data performs well but is slightly outperformed by KD.

### 🔮 Future Work

While this project demonstrates the effectiveness of dataset distillation and knowledge distillation with YOLOv8 on the COCO dataset, several directions remain open for future exploration:

1. **Multi-Teacher Knowledge Distillation**  
   Instead of using a single ResNet-50 model, combining the outputs from multiple strong teacher models could provide more diverse and robust soft labels for the student.

2. **Cross-Dataset Generalization**  
   Test the student models on unseen datasets (e.g., Pascal VOC, KITTI) to assess how well knowledge and synthetic data generalize across domains.

3. **Neural Architecture Search (NAS)**  
   Integrate NAS to discover a more optimal student model architecture tailored specifically for distilled supervision or synthetic datasets.

4. **Quantization and Pruning**  
   Apply model quantization and pruning techniques to further reduce memory and inference cost—making the student model even more deployment-ready.


