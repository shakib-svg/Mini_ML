#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 00:39:18 2025

Adapted by CLASSIFICATION on
train/ and test/ directories each containing one subfolder per class
with 100×100 PNG spectrogram images.
Outputs F2 score, classification report, and confusion matrix.
"""

import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_dataset(root_dir):
    """
    Load images and labels from a directory structured as:
      root_dir/
        class1/*.png
        class2/*.png
    Returns:
      X: (n_samples, n_features) numpy array of flattened RGB images
      y: (n_samples,) numpy array of integer labels
      labels: list of class names, in the order of their indices
    """
    X, y = [], []
    # discover class subfolders, sorted for consistent label order
    labels = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        folder = os.path.join(root_dir, label)
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.png'):
                continue
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img, dtype=np.uint8)
            X.append(arr.flatten())
            y.append(label_to_idx[label])

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=int)
    return X, y, labels

def plot_confusion_matrix(cm, labels):
    """
    Display a confusion matrix with matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():
    # Paths to your train/test folders
    train_dir = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/final_train/train'
    test_dir  = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/final_train/test'

    # 1) Load datasets
    print("Loading training data from:", train_dir)
    X_train, y_train, labels = load_dataset(train_dir)
    print(f"  → {X_train.shape[0]} samples, {len(labels)} classes")

    print("Loading test data from:", test_dir)
    X_test, y_test, _ = load_dataset(test_dir)
    print(f"  → {X_test.shape[0]} samples")

    # 2) Shuffle training set
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # 3) Initialize & train classifier
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    print("Training XGBoost classifier...")
    clf.fit(X_train, y_train)

    # 4) Predict on test set
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)

    # 5) Compute F2 score (macro-average)
    f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
    print(f"\nF2 score (macro-average): {f2:.4f}")

    # 6) Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    # 7) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels)

if __name__ == "__main__":
    main()