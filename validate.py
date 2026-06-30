import os
import time
import numpy as np
from predict import predict

# Folders containing the dataset (for validation)
REAL_DIR = "dataset/train/real"
SCREEN_DIR = "dataset/train/screen"

# Exclude identical files to ensure clean validation
DUPLICATES_TO_EXCLUDE = {
    "real_035.jpeg", "screen_009.jpeg",
    "real_038.jpeg", "screen_034.jpeg",
    "real_043.jpeg", "screen_38.jpeg"
}

def validate():
    reals = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in DUPLICATES_TO_EXCLUDE]
    screens = [os.path.join(SCREEN_DIR, f) for f in os.listdir(SCREEN_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in DUPLICATES_TO_EXCLUDE]
    
    all_paths = reals + screens
    y_true = [0] * len(reals) + [1] * len(screens)
    
    print(f"Validating {len(all_paths)} images...")
    
    predictions = []
    latencies = []
    
    for i, path in enumerate(all_paths):
        t0 = time.perf_counter()
        prob = predict(path)
        t1 = time.perf_counter()
        
        # Latency in milliseconds
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)
        
        # Hard threshold of 0.5 for binary classification
        pred = 1 if prob >= 0.5 else 0
        predictions.append(pred)
        
        # Output progress every 20 images
        if (i + 1) % 20 == 0 or (i + 1) == len(all_paths):
            print(f"Processed {i + 1}/{len(all_paths)} images...")
            
    # Calculate metrics
    predictions = np.array(predictions)
    y_true = np.array(y_true)
    
    accuracy = np.mean(predictions == y_true) * 100.0
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    
    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)
    print(f"Total Evaluated Images : {len(all_paths)}")
    print(f"Final Accuracy         : {accuracy:.2f}%")
    print(f"Average Latency        : {avg_latency:.2f} ms per image")
    print(f"Median Latency         : {median_latency:.2f} ms per image")
    print(f"Latency Std Dev        : {std_latency:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    validate()
