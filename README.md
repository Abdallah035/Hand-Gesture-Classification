# Hand Gesture Classification

A machine learning project that classifies **18 different hand gestures** using hand landmark coordinates extracted with **MediaPipe**. The project includes model training, evaluation, MLflow experiment tracking, model registry, and real-time video inference.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Hand Gestures](#hand-gestures)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training & Results](#model-training--results)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [MLflow Model Registry](#mlflow-model-registry)
- [Video Inference](#video-inference)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)

---

## Overview

This project uses **21 hand landmarks** (x, y, z coordinates) detected by MediaPipe to classify hand gestures. Three models were trained and compared using MLflow for experiment tracking.

**Pipeline:**
1. Extract hand landmarks from images using MediaPipe
2. Normalize landmarks (center on wrist, scale by middle finger)
3. Train multiple classifiers (Random Forest, KNN, Logistic Regression)
4. Track experiments with MLflow
5. Register best model in MLflow Model Registry
6. Run real-time gesture recognition on video

---

## Dataset

- **Source:** Hand landmarks extracted using MediaPipe HandLandmarker
- **Samples:** 25,675
- **Features:** 63 (x, y, z for 21 landmarks)
- **Classes:** 18 gesture types
- **File:** `hand_landmarks_data.csv`

---

## Hand Gestures

| Gesture | Gesture | Gesture |
|---------|---------|---------|
| call | dislike | fist |
| four | like | mute |
| ok | one | palm |
| peace | peace_inverted | rock |
| stop | stop_inverted | three |
| three2 | two_up | two_up_inverted |

---

## Project Structure

```
Hand Gesture Classification/
│
├── Hand-Gesture.ipynb          # Main notebook (training, evaluation, MLflow)
├── mlflow_utils.py             # MLflow helper functions
├── hand_landmarks_data.csv     # Dataset
├── best_model.pkl              # Saved best model (Random Forest)
├── label_encoder.pkl           # Saved label encoder
├── hand_landmarker.task        # MediaPipe hand landmarker model
├── test_abdallah2.mp4          # Input test video
├── output_video.avi            # Output video with predictions
├── mlflow.db                   # MLflow tracking database
├── mlruns/                     # MLflow artifacts
├── screenshots/                # Screenshots for documentation
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Hand-Gesture-Classification.git
cd Hand-Gesture-Classification
```

### 2. Create virtual environment (Python 3.12)

```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1    # PowerShell
# or
source venv/Scripts/activate   # Git Bash
```

### 3. Install dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn mediapipe opencv-python joblib scipy mlflow
```

---

## Usage

### Run the notebook

```bash
jupyter notebook Hand-Gesture.ipynb
```

### Launch MLflow UI

```bash
mlflow ui
```

Then open: http://127.0.0.1:5000

---

## Model Training & Results

Three models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **96.51%** | **96.55%** | **96.51%** | **96.52%** |
| KNN | 93.01% | 93.08% | 93.01% | 93.02% |
| Logistic Regression | 86.31% | 86.28% | 86.31% | 86.23% |

**Best Model:** Random Forest with **96.51% accuracy**

---

## MLflow Experiment Tracking

All experiments are tracked using MLflow with:

- **Experiment Name:** `Hand-Gesture-Classification`
- **Per Run Logging:**
  - Dataset info (rows, features, classes)
  - Model hyperparameters
  - Metrics (accuracy, precision, recall, F1-score)
  - Trained model artifact with signature
  - Classification report
  - Confusion matrix
- **Comparison Chart:** Bar chart comparing all models across all metrics

---

## MLflow Model Registry

The best model (Random Forest) is registered in the MLflow Model Registry:

- **Registered Model Name:** `Hand-Gesture-Classifier`
- **Alias:** `champion` (production-ready)
- **Description:** Random Forest (n_estimators=100) trained on 20,540 samples

Load the model from registry:

```python
import mlflow
model = mlflow.sklearn.load_model("models:/Hand-Gesture-Classifier@champion")
```

---

## Video Inference

The project processes video frame-by-frame:

1. Detect hand landmarks using MediaPipe
2. Normalize landmarks (same as training)
3. Predict gesture using the best model
4. Stabilize prediction with a sliding window (size=15)
5. Draw skeleton and label on each frame
6. Save output video

---

## Screenshots

> Add screenshots of:
> - MLflow experiment runs
> - MLflow charts and metrics comparison
> - MLflow Model Registry with versions
> - Model comparison chart
> - Sample gesture predictions

Place screenshots in the `screenshots/` folder.

---

## Technologies Used

- **Python 3.12**
- **scikit-learn** — Model training (Random Forest, KNN, Logistic Regression)
- **MediaPipe** — Hand landmark detection
- **OpenCV** — Video processing
- **MLflow** — Experiment tracking & Model Registry
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
