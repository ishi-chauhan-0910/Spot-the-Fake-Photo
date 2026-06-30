# Spot the Fake Photo

## Overview

This project classifies an input image as either a **Real Photo** or a **Photo of a Screen**. The solution is implemented using classical computer vision techniques and a Support Vector Machine (SVM) classifier. Image features are extracted from each input image and used to predict the probability that the image is a recaptured screen photo.

---

## Project Structure

```
ML_Latest_Assignment/
│
├── dataset/
│   └── train/
│       ├── real/
│       └── screen/
│
├── templates/
│   └── index.html
│
├── app.py
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

A custom dataset was prepared for this assignment.

The dataset contains:

- 481 real photographs
- 110 photographs of screens

The images include different lighting conditions, viewing angles, distances, and devices to improve the model's ability to generalize.

---

## Requirements

Install the required Python packages:

```bash
pip install numpy opencv-python pillow scikit-learn flask joblib scipy
```

---

## Training

Train the classifier using:

```bash
python train.py
```

Example output:

```
Validation Accuracy: 98.32%
Model trained successfully and saved to: model.pkl
```

---

## Validation

Run:

```bash
python validate.py
```

Validation results:

- Total Images Evaluated: **591**
- Final Accuracy: **98.82%**
- Average Latency: **620.14 ms per image**
- Median Latency: **602.01 ms per image**

---

## Prediction

Predict a single image:

```bash
python predict.py image.jpg
```

Example output:

```
0.94
```

Output interpretation:

- Score close to **0** → Real Photo
- Score close to **1** → Screen Photo

---

## Web Interface

Start the Flask application:

```bash
python app.py
```

Open the URL displayed in the terminal to upload and classify an image.

---

## Approach

The implementation uses handcrafted image features including:

- RGB color statistics
- HSV color statistics
- Laplacian sharpness
- Frequency-domain features using FFT

These features are normalized and passed to an SVM classifier with an RBF kernel to generate a probability score.
