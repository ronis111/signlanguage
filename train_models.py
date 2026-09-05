"""
Unified Model Training and Export Script for Sign Language Detection.

Algorithms implemented:
1. Random Forest (RF) - Ensemble of decision trees
2. Support Vector Machine (SVM) - RBF Kernel Hyperplane Classifier
3. Convolutional Neural Network (CNN) - 1D-CNN on normalized landmarks
4. Benchmarking models: Multi-Layer Perceptron (MLP), K-Nearest Neighbors (KNN)

Preprocessing:
- Wrist-relative translation: (x_i - x_0, y_i - y_0, z_i - z_0)
- Scale normalization: normalized by distance between wrist (0) and middle finger MCP (9)
This eliminates position bias (e.g. 'A' mispredicted as 'G'/'H' when hand is in center).
"""

import json
import os
import shutil
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from keras import layers, models
from keras.callbacks import EarlyStopping


def normalize_landmarks(X_raw):
    """
    Normalizes 21 3D landmarks (63 values) to be wrist-relative and scale-invariant.
    Supports input shape (63,), (21, 3), (N, 63), or (N, 21, 3).
    """
    is_1d = False
    arr = np.array(X_raw, dtype=np.float32)
    if arr.ndim == 1:
        is_1d = True
        arr = arr.reshape(1, 21, 3)
    elif arr.ndim == 2 and arr.shape[1] == 63:
        arr = arr.reshape(-1, 21, 3)
    elif arr.ndim == 2 and arr.shape == (21, 3):
        is_1d = True
        arr = arr.reshape(1, 21, 3)

    N = arr.shape[0]
    X_norm = np.zeros((N, 63), dtype=np.float32)

    for i in range(N):
        pts = arr[i].copy()
        wrist = pts[0].copy()  # Landmark 0: wrist
        rel = pts - wrist     # Translate relative to wrist

        # Scale by Euclidean distance between wrist (0) and middle finger MCP (9)
        scale = np.linalg.norm(rel[9])
        if scale == 0:
            scale = np.max(np.abs(rel))
        if scale > 0:
            rel = rel / scale

        X_norm[i] = rel.flatten()

    return X_norm[0] if is_1d else X_norm


def train_all_models():
    print("=" * 70)
    print("SIGN LANGUAGE DETECTION: MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    # 1. Load Dataset
    data_path = 'dataset.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"\n1. Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(str).values

    print(f"   Total samples: {len(X)}")
    print(f"   Raw features per sample: {X.shape[1]}")
    unique_classes = sorted(list(np.unique(y)))
    print(f"   Classes ({len(unique_classes)}): {unique_classes}")

    # 2. Normalize Landmarks
    print("\n2. Applying wrist-relative scale-invariant normalization...")
    X_norm = normalize_landmarks(X)

    # 3. Train/Test Split
    print("\n3. Splitting dataset (80% Train, 20% Test, Stratified)...")
    x_train, x_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train samples: {len(x_train)}")
    print(f"   Test samples:  {len(x_test)}")

    results = {}

    # -------------------------------------------------------------------------
    # 4. RANDOM FOREST CLASSIFIER
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = rf_acc
    print(f"   Random Forest Test Accuracy: {rf_acc * 100:.2f}%")
    print("\n   Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=unique_classes))

    # -------------------------------------------------------------------------
    # 5. SUPPORT VECTOR MACHINE (SVM) CLASSIFIER
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("5. Training Support Vector Machine (SVM) Classifier (RBF Kernel)...")
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm_model.fit(x_train, y_train)
    svm_pred = svm_model.predict(x_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    results['Support Vector Machine (SVM)'] = svm_acc
    print(f"   SVM Test Accuracy: {svm_acc * 100:.2f}%")
    print("\n   SVM Classification Report:")
    print(classification_report(y_test, svm_pred, target_names=unique_classes))

    # -------------------------------------------------------------------------
    # 6. CONVOLUTIONAL NEURAL NETWORK (CNN / Conv1D)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("6. Retraining CNN (Conv1D) on Normalized Landmarks...")

    lb = LabelBinarizer()
    y_train_onehot = lb.fit_transform(y_train)
    y_test_onehot = lb.transform(y_test)

    x_train_cnn = x_train.reshape(-1, 63, 1)
    x_test_cnn = x_test.reshape(-1, 63, 1)

    cnn_model = models.Sequential([
        layers.Input(shape=(63, 1)),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(unique_classes), activation='softmax')
    ])

    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )

    cnn_model.fit(
        x_train_cnn,
        y_train_onehot,
        epochs=35,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    cnn_eval_loss, cnn_acc = cnn_model.evaluate(x_test_cnn, y_test_onehot, verbose=0)
    results['CNN (Conv1D)'] = cnn_acc
    print(f"   CNN Test Accuracy: {cnn_acc * 100:.2f}%")

    cnn_pred_probs = cnn_model.predict(x_test_cnn, verbose=0)
    cnn_pred = lb.classes_[np.argmax(cnn_pred_probs, axis=1)]
    print("\n   CNN Classification Report:")
    print(classification_report(y_test, cnn_pred, target_names=unique_classes))

    # -------------------------------------------------------------------------
    # 7. BENCHMARK COMPARISONS (MLP & KNN)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("7. Benchmarking Additional Candidate Algorithms...")

    # Multi-Layer Perceptron (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
    mlp.fit(x_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(x_test))
    results['Multi-Layer Perceptron (MLP)'] = mlp_acc
    print(f"   MLP Test Accuracy: {mlp_acc * 100:.2f}%")

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(x_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(x_test))
    results['K-Nearest Neighbors (KNN)'] = knn_acc
    print(f"   KNN Test Accuracy: {knn_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # 8. POSITION-INVARIANCE TEST (Testing 'A' shifted to center screen)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. POSITION-INVARIANCE TEST: Verifying 'A' held in center/left of screen")
    print("=" * 70)

    a_raw = df[df[63] == 'A'].iloc[:20, :63].values
    shifted_a = a_raw.copy().reshape(-1, 21, 3)
    # Shift wrist from ~0.75 to 0.50 (center) and y from 0.57 to 0.72 (lower center)
    shifted_a[:, :, 0] += (0.50 - shifted_a[:, 0, 0].mean())
    shifted_a[:, :, 1] += (0.72 - shifted_a[:, 0, 1].mean())
    shifted_a_norm = normalize_landmarks(shifted_a.reshape(-1, 63))

    rf_shifted_preds = rf_model.predict(shifted_a_norm)
    svm_shifted_preds = svm_model.predict(shifted_a_norm)
    cnn_shifted_preds = lb.classes_[np.argmax(cnn_model.predict(shifted_a_norm.reshape(-1, 63, 1), verbose=0), axis=1)]

    print(f"   Random Forest predictions for centered 'A': {list(rf_shifted_preds)}")
    print(f"   SVM predictions for centered 'A':           {list(svm_shifted_preds)}")
    print(f"   Retrained CNN predictions for centered 'A': {list(cnn_shifted_preds)}")

    all_rf_correct = all(p == 'A' for p in rf_shifted_preds)
    all_svm_correct = all(p == 'A' for p in svm_shifted_preds)
    all_cnn_correct = all(p == 'A' for p in cnn_shifted_preds)

    if all_rf_correct and all_svm_correct and all_cnn_correct:
        print("   SUCCESS: Position-invariance verified! 'A' is 100% correctly predicted without 'G'/'H' false detections.")
    else:
        print("   WARNING: Some models misclassified shifted 'A'.")

    # -------------------------------------------------------------------------
    # 9. EXPORT MODELS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. Exporting Models...")
    print("=" * 70)

    static_hdf = os.path.join('static', 'hdf')
    os.makedirs(static_hdf, exist_ok=True)

    # Export Random Forest
    joblib.dump(rf_model, 'random_forest_model.joblib')
    joblib.dump(rf_model, os.path.join(static_hdf, 'random_forest_model.joblib'))
    print("   Saved random_forest_model.joblib to root and static/hdf/")

    # Export SVM
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(svm_model, os.path.join(static_hdf, 'svm_model.joblib'))
    print("   Saved svm_model.joblib to root and static/hdf/")

    # Export CNN
    cnn_model.save('cnn_model_final2.h5')
    cnn_model.save(os.path.join(static_hdf, 'cnn_model_final1.h5'))
    print("   Saved cnn_model_final2.h5 to root and static/hdf/cnn_model_final1.h5")

    # Export Label Map
    label_map = {int(i): c for i, c in enumerate(unique_classes)}
    with open('label_classes.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    with open(os.path.join(static_hdf, 'label_classes.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    print("   Saved label_classes.json to root and static/hdf/")

    # -------------------------------------------------------------------------
    # 10. SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<35} | {'Test Accuracy':<15}")
    print("-" * 55)
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<35} | {acc * 100:.2f}%")
    print("=" * 70)
    print("Training and export complete!\n")


if __name__ == '__main__':
    train_all_models()
