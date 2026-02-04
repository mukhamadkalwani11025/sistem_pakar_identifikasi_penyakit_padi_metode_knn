import os
import cv2
import numpy as np
import pandas as pd

# Path ke folder dataset
DATASET_PATH = "rice_leaf_diseases"

data = []

# Loop setiap folder kelas
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)

        # Baca gambar
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize gambar
        img = cv2.resize(img, (128, 128))

        # Konversi BGR ke RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Hitung mean RGB
        r_mean = np.mean(img[:, :, 0])
        g_mean = np.mean(img[:, :, 1])
        b_mean = np.mean(img[:, :, 2])

        data.append([r_mean, g_mean, b_mean, label])

# Buat DataFrame
df = pd.DataFrame(data, columns=["r_mean", "g_mean", "b_mean", "label"])

# Simpan ke CSV
df.to_csv("dataset_padi_rgb.csv", index=False)

print("✅ Ekstraksi fitur selesai!")
print(df.head())
