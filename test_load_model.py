import joblib

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

print(model)
print(scaler)
