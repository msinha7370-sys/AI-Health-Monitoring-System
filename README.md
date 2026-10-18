# Healthcare Disease Diagnosis Predictor

This repository contains a Machine Learning pipeline that predicts potential medical conditions based on user-reported symptoms. It trains a Random Forest Classifier on a comprehensive multi-symptom dataset and provides an interactive UI inside a Jupyter/Colab notebook for symptom selection and inference.

## 📊 Dataset

Due to GitHub's file size limitations (files exceeding 25 MB cannot be uploaded directly or efficiently tracked without Git LFS), the raw **`dataset.csv`** is not hosted in this repository. 

* **Download Source:** You can download the dataset directly from **Kaggle** (or your specific Kaggle dataset link).
* **Filename:** `dataset.csv`
* **Structure:** The dataset contains 246,945 records tracking 377 unique symptoms as binary features (`0` or `1`) mapped to a target `diseases` column.

### Setup Instructions for Dataset
1. Download `dataset.csv` from Kaggle.
2. Place the `dataset.csv` file in the root directory of this project (the same folder as the notebook) before executing the cells.

---

## 🚀 Project Features

- **Efficient Training Pipeline:** Uses Scikit-Learn's `RandomForestClassifier` for multi-class classification over hundreds of diagnostic targets.
- **Serialization:** Saves the trained model (`healthcare_rf_model.pkl`) and training features (`symptom_columns.pkl`) using `joblib` for production deployment.
- **Interactive UI:** Features an in-notebook `ipywidgets` GUI allowing users to dynamically search, filter, and select multiple symptoms to obtain real-time diagnoses along with model confidence percentages.

---

## 🛠️ Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install numpy pandas scikit-learn joblib ipywidgets
2. Running the Notebook
Clone this repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Place the downloaded dataset.csv into the directory.
Launch Jupyter Notebook or upload the project to Google Colab, and run all cells.
🧠 Model Performance
Model Type: Random Forest Classifier
Accuracy: ~82.23%
Inputs: 377 symptom checkpoints
Outputs: Top 3 predicted medical conditions with probability scores
