"""Spot the Fake Photo predictor.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
from scipy import ndimage


def extract_features(image_path: str) -> list:
    """Extract 33 statistical, sharpness, and frequency features from an image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # 1. Color features (12 features)
    arr = np.array(img, dtype=np.float32)
    r_chan, g_chan, b_chan = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    features = [
        np.mean(r_chan), np.std(r_chan),
        np.mean(g_chan), np.std(g_chan),
        np.mean(b_chan), np.std(b_chan),
    ]
    
    hsv_img = img.convert("HSV")
    hsv_arr = np.array(hsv_img, dtype=np.float32)
    h_chan, s_chan, v_chan = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]
    
    features.extend([
        np.mean(h_chan), np.std(h_chan),
        np.mean(s_chan), np.std(s_chan),
        np.mean(v_chan), np.std(v_chan),
    ])
    
    # 2. Sharpness features (3 features)
    gray = img.convert("L")
    gray_arr = np.array(gray, dtype=np.float32)
    
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian = ndimage.convolve(gray_arr, laplacian_kernel)
    
    features.extend([
        np.var(laplacian),
        np.mean(np.abs(laplacian)),
        np.max(np.abs(laplacian))
    ])
    
    # 3. FFT Frequency features (18 features)
    crop_size = 256
    if w >= crop_size and h >= crop_size:
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        patch = gray.crop((left, top, left + crop_size, top + crop_size))
    else:
        patch = gray.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
        
    patch_arr = np.array(patch, dtype=np.float32)
    
    f_coef = np.fft.fft2(patch_arr)
    f_shift = np.fft.fftshift(f_coef)
    magnitude = np.abs(f_shift)
    magnitude = np.log(magnitude + 1e-9)
    
    cy, cx = crop_size // 2, crop_size // 2
    y, x = np.ogrid[-cy:crop_size-cy, -cx:crop_size-cx]
    r = np.sqrt(x**2 + y**2)
    
    bands = [(0, 15), (15, 30), (30, 60), (60, 90), (90, 128), (128, 180)]
    for low, high in bands:
        mask = (r >= low) & (r < high)
        if np.sum(mask) > 0:
            band_vals = magnitude[mask]
            features.extend([
                np.mean(band_vals),
                np.std(band_vals),
                np.max(band_vals)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
            
    return features


def predict(image_path: str) -> float:
    # Resolve model.pkl absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file 'model.pkl' not found at {model_path}. Run train.py first.")
        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    mean = model_data["mean"]
    std = model_data["std"]
    model = model_data["model"]
    
    # Extract features
    feats = extract_features(image_path)
    feats_arr = np.array(feats, dtype=np.float32).reshape(1, -1)
    
    # Scale features
    feats_scaled = (feats_arr - mean) / std
    
    # Predict probability of being a screen recapture (class 1)
    prob = model.predict_proba(feats_scaled)[0][1]
    return float(prob)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    print(predict(sys.argv[1]))
