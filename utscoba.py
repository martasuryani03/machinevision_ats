import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import LeaveOneOut
from skimage.feature import hog
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# === CEK FILE DAN DATASET ===
print("Current directory:", os.getcwd())
files = os.listdir()
print("Files in folder:", files)

# === LOAD DATASET EMNIST (CSV) ===
csv_path = "emnist-letters-train.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("File emnist-letters-train.csv tidak ditemukan!")

data = pd.read_csv(csv_path)
print("Dataset terbaca dengan bentuk:", data.shape)

# Pisahkan fitur dan label
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
print("Jumlah kelas unik:", len(np.unique(y)))

# === SAMPLING DATA SEIMBANG (26 kelas x 500 = 13.000) ===
samples_per_class = 500
X_balanced, y_balanced = [], []
for label in np.unique(y):
    idx = np.where(y == label)[0][:samples_per_class]
    X_balanced.append(X[idx])
    y_balanced.append(y[idx])
X_balanced = np.vstack(X_balanced)
y_balanced = np.hstack(y_balanced)
print(f"Data setelah sampling: {X_balanced.shape} Label: {y_balanced.shape}")

# === EKSTRAKSI FITUR HOG ===
print("\nMengekstraksi fitur HOG (Histogram of Oriented Gradients)...")
hog_features = []
for img in tqdm(X_balanced.reshape(-1, 28, 28), desc="HOG Extraction"):
    features = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    hog_features.append(features)
hog_features = np.array(hog_features)
print("Dimensi fitur HOG:", hog_features.shape)

# === MODEL SVM ===
clf = SVC(kernel='rbf', C=10, gamma='scale')

# === EVALUASI DENGAN LEAVE-ONE-OUT CROSS VALIDATION (LOOCV) ===
print("\nMelakukan evaluasi dengan Leave-One-Out Cross Validation (LOOCV)...")
loo = LeaveOneOut()

y_true, y_pred = [], []
for train_index, test_index in tqdm(loo.split(hog_features), total=len(hog_features), desc="LOOCV"):
    X_train, X_test = hog_features[train_index], hog_features[test_index]
    y_train, y_test = y_balanced[train_index], y_balanced[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    y_true.append(y_test[0])
    y_pred.append(pred[0])

# === HASIL EVALUASI ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("\n=== HASIL EVALUASI LOOCV ===")
print(f"Akurasi  : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"F1-score : {f1:.4f}")

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix – SVM EMNIST Letters (LOOCV)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === SIMPAN LAPORAN DAN MODEL ===
report = classification_report(y_true, y_pred, digits=3)
with open("hasil_evaluasi_loocv.txt", "w") as f:
    f.write("=== Hasil Evaluasi LOOCV ===\n")
    f.write(report)
print("\nLaporan evaluasi tersimpan di 'hasil_evaluasi_loocv.txt'")

dump(clf, "svm_emnist_model_loocv.pkl")
print("Model tersimpan ke 'svm_emnist_model_loocv.pkl'")

print("\nProgram selesai dijalankan.")
