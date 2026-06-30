# Spot the Fake Photo – Project Report

## Problem Statement

The objective of this project is to identify whether an input image is a genuine photograph captured directly using a camera or a photograph of a screen displaying an image.

The solution is intended to detect recaptured images using classical computer vision techniques without relying on deep learning.

---

## Dataset

A custom dataset was created for this project.

The dataset consists of two categories:

- Real photographs captured using mobile devices.
- Photographs of mobile and laptop screens displaying images.

The final training dataset contains:

- 481 real photographs
- 110 screen photographs

Images were collected under different lighting conditions, viewing angles, distances, and screen brightness levels to improve robustness.

---

## Methodology

The solution follows a classical machine learning pipeline.

For each image, multiple handcrafted features are extracted, including:

- RGB color statistics
- HSV color statistics
- Laplacian-based sharpness
- Frequency-domain characteristics using Fast Fourier Transform (FFT)

The extracted feature vector is normalized before training.

A Support Vector Machine (SVM) with an RBF kernel is trained using these feature vectors.

During prediction, the same feature extraction process is applied to the input image, and the trained classifier produces a probability score between 0 and 1.

---

## Model

**Classifier:** Support Vector Machine (SVM)

**Kernel:** Radial Basis Function (RBF)

**Regularization Parameter (C):** 1.0

---

## Results

### Training

- Training Images: 591
- Validation Accuracy During Training: **98.32%**

### Validation

- Total Evaluated Images: **591**
- Final Accuracy: **98.82%**
- Average Inference Latency: **620.14 ms per image**
- Median Inference Latency: **602.01 ms per image**
- Latency Standard Deviation: **683.18 ms**

Testing was performed on a Windows laptop using CPU-based inference.

---

## Cost Per Image

The model performs inference locally without requiring cloud resources.

Therefore, the operational cost per image is effectively zero for local deployment.

---

## Future Improvements

- Increase the size and diversity of the training dataset.
- Include additional device types and display technologies.
- Evaluate the model on larger independent benchmark datasets.
- Compare the classical approach with lightweight deep learning models such as MobileNetV3.

---

## Conclusion

The proposed solution successfully distinguishes real photographs from photographs of screens using handcrafted image features and an SVM classifier.

The final model achieved **98.82% validation accuracy** while maintaining efficient CPU-based inference. The approach is lightweight, easy to deploy, and does not require GPU hardware or cloud services.