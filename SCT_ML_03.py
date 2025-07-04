import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
# Parameters
IMAGE_SIZE = 64
CAT_LABEL = 0
DOG_LABEL = 1
CAT_DIR = r"C:\Users\arunp\Downloads\Cat"
DOG_DIR = r"C:\Users\arunp\Downloads\Dog"
# Load Images
def load_images(folder, label):
    data, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            data.append(img.flatten())
            labels.append(label)
    return data, labels
# Loading and combining data
print("Loading images...")
cat_data, cat_labels = load_images(CAT_DIR, CAT_LABEL)
dog_data, dog_labels = load_images(DOG_DIR, DOG_LABEL)
X = np.array(cat_data + dog_data)
y = np.array(cat_labels + dog_labels)
# Print class distribution
cat_count = sum(np.array(y) == 0)
dog_count = sum(np.array(y) == 1)
print(f"\nDataset Summary:")
print(f"Total images: {len(y)}")
print(f"Cat images: {cat_count}")
print(f"Dog images: {dog_count}")
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
# Train SVM
print("\nTraining SVM...")
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)
# Predict & Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Cat", "Dog"], output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
misclassification_rate = 1 - acc
# Print metrics
print("\n--- Accuracy ---")
print(f"Accuracy: {acc:.4f}")
print(f"Misclassification Rate: {misclassification_rate:.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Accuracy Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Misclassification"], [acc, misclassification_rate], color=['green', 'red'])
plt.ylim(0, 1)
plt.title("Model Performance")
plt.ylabel("Score")
plt.show()
# Class Distribution Pie Chart
plt.figure(figsize=(5, 5))
plt.pie([cat_count, dog_count], labels=["Cats", "Dogs"], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title("Class Distribution in Dataset")
plt.show()
