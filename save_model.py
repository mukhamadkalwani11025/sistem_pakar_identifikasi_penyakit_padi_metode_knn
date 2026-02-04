import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("dataset_padi_rgb.csv")

X = df[['r_mean', 'g_mean', 'b_mean']]
y = df['label']

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model KNN (pakai K terbaik)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Simpan model & scaler
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model KNN dan Scaler berhasil disimpan")
