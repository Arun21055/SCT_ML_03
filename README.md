# 🐱🐶 Cat vs Dog Image Classifier using SVM

This project is a **binary image classification** model that distinguishes between **cats and dogs** using **Support Vector Machines (SVM)** and the **Kaggle Cats vs Dogs dataset**. It includes data preprocessing, model training, evaluation, and visualization of statistical outputs and results.

---

## 📁 Dataset

- Source: [Microsoft Dogs vs Cats Dataset (Kaggle)](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- Classes:
  - **Cat** → Label `0`
  - **Dog** → Label `1`
- Input: Grayscale images resized to `64x64` pixels for simplicity.
- Some corrupted images are automatically skipped during preprocessing.

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 Features

- 📷 Load and preprocess images from local folders  
- 🔍 Train/Test split with `stratify` for class balance  
- 🧠 Trains an SVM model with linear kernel  
- 📊 Evaluates model with:
  - Accuracy score
  - Classification report
  - Confusion matrix
  - Misclassification rate
- 📈 Visualizations:
  - Confusion matrix heatmap
  - Accuracy vs Misclassification bar chart
  - Class distribution pie chart
- 🔮 Predicts the class (cat/dog) for a new input image

---

## 🧪 How to Run

1. Clone this repository  
2. Install dependencies:

   ```bash
   pip install numpy opencv-python scikit-learn matplotlib seaborn
