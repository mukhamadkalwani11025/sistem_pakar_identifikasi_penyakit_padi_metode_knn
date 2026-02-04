import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset_padi_rgb.csv")

X = df[['r_mean', 'g_mean', 'b_mean']]
y = df['label']

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data (biar konsisten, random_state sama)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

k_values = [3, 5, 7, 9]
results = []

print("Hasil pengujian nilai K:\n")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append((k, acc))

    print(f"K = {k} → Akurasi = {acc:.4f}")

# Tampilkan K terbaik
best_k = max(results, key=lambda x: x[1])
print("\nK terbaik:")
print(f"K = {best_k[0]} dengan akurasi = {best_k[1]:.4f}")
