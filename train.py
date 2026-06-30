import os
import pickle
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Folders containing the dataset
REAL_DIR = "dataset/train/real"
SCREEN_DIR = "dataset/train/screen"

# Duplicates to exclude to avoid label leakage / noise
DUPLICATES_TO_EXCLUDE = {
    "real_035.jpeg", "screen_009.jpeg",
    "real_038.jpeg", "screen_034.jpeg",
    "real_043.jpeg", "screen_38.jpeg"
}

def extract_features(img_path: str) -> list:
    """Extract 33 statistical, sharpness, and frequency features from an image."""
    img = Image.open(img_path).convert("RGB")
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

def train():
    print("Preparing training dataset...")
    reals = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in DUPLICATES_TO_EXCLUDE]
    screens = [os.path.join(SCREEN_DIR, f) for f in os.listdir(SCREEN_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in DUPLICATES_TO_EXCLUDE]
    
    all_paths = reals + screens
    y = np.array([0] * len(reals) + [1] * len(screens), dtype=np.int32)
    
    print(f"Loaded {len(reals)} real photos and {len(screens)} screen recaptures.")
    print("Extracting features (this might take a few seconds)...")
    
    X = np.array([extract_features(p) for p in all_paths], dtype=np.float32)
    
    print("Splitting dataset into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate feature scaling factors (mean and standard deviation) on training data ONLY
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    
    print("Training SVM classifier (RBF kernel, C=1.0)...")
    # Enable probability estimation for outputting score in [0, 1]
    model = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    
    # Save the scaler parameters and the model to a single pickle file
    model_data = {
        "mean": mean,
        "std": std,
        "model": model
    }
    
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"Model trained successfully and saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    train()
