# Spot the Fake Photo

## Overview

This project detects whether an input image is a **Real Photo** or a **Photo of a Screen (Recaptured Image)**. The solution uses classical computer vision techniques and a Support Vector Machine (SVM) classifier. A set of handcrafted image features is extracted from each input image, and the trained model predicts the probability of the image being a screen recapture.

---

## Project Structure

```
ML_Latest_Assignment/
│
├── train.py
├── validate.py
├── predict.py
├── model.pkl
├── README.md
├── report.md
└── ASSIGNMENT.pdf
```

---

## Dataset

The training dataset is provided separately due to GitHub file size limitations.

### Dataset Download

Google Drive:

https://drive.google.com/drive/folders/1tl5ZSPrpySvGB5MoQXB8cDPMO988rCKq?usp=sharing

After downloading the dataset, place it in the following directory structure:

```
dataset/
├── screen/
└── real/
```

The dataset contains real photographs and photographs of screens collected under different lighting conditions, viewing angles, distances, and devices to improve the model's ability to generalize.

---

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy pillow scikit-learn opencv-python joblib
```

---

## Training

Train the model using:

```bash
python train.py
```

Example output:

```
Preparing training dataset...
Training SVM classifier...
Model trained successfully and saved to: model.pkl
```

---

## Validation

Run:

```bash
python validate.py
```

Example validation results:

```
========================================
VALIDATION RESULTS
========================================
Total Evaluated Images : 591
Final Accuracy         : 98.82%
Average Latency        : 620.14 ms per image
Median Latency         : 602.01 ms per image
Latency Std Dev        : 683.18 ms
========================================
```

---

## Prediction

Predict a single image using:

```bash
python predict.py image.jpg
```

Example output:

```
0.94
```

### Output Interpretation

- **Score close to 0** → Real Photo
- **Score close to 1** → Photo of a Screen

---

## Methodology

The implementation extracts a 33-dimensional feature vector from each image using:

- RGB color statistics
- HSV color statistics
- Laplacian-based sharpness features
- Fast Fourier Transform (FFT) frequency features

The extracted features are normalized and passed to a Support Vector Machine (SVM) with an RBF kernel to generate a probability score between **0** and **1**.

---

## Notes

- The pretrained model (`model.pkl`) is included in this repository.
- The dataset is provided through the Google Drive link above due to GitHub storage limitations.
- The project has been tested on Windows using Python 3.11.
