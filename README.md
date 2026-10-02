# Industrial Machine Failure Prediction

### Learn Depth Academy — Track 1 Capstone | Problem 21

A machine-learning classification project that predicts whether an industrial machine is likely to experience a failure based on operating and maintenance-related data.

**Domain:** Industrial Systems
**ML Task:** Binary Classification
**Dataset:** AI4I 2020 Predictive Maintenance Dataset
**Application:** Streamlit ML Prediction Prototype
**Status:** In Development

---

## 📌 Overview

Unexpected machine failures can lead to production downtime, increased maintenance costs, and operational losses.

This project explores how machine-learning classification techniques can be used to identify patterns associated with machine failures and provide an early prediction of failure risk.

The project implements and compares multiple classification models and evaluates them using metrics that are particularly relevant to failure detection.

---

## 🎯 Objectives

* Predict whether a machine failure is likely to occur.
* Explore machine operating and maintenance data.
* Preprocess and prepare the dataset for machine learning.
* Train multiple binary classification models.
* Compare model performance using appropriate evaluation metrics.
* Save the trained model and evaluation outputs.
* Build an interactive Streamlit application for predictions.

---

## 📊 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the **UCI Machine Learning Repository**.

**Dataset:** AI4I 2020 Predictive Maintenance Dataset
**Source:** UCI Machine Learning Repository
**Observations:** 10,000

The dataset contains machine-related variables such as:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine failure information

The dataset is synthetic, but it was designed to represent predictive-maintenance scenarios encountered in industrial environments.

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### Dataset Handling

The dataset is **not stored directly in this repository**.

It is retrieved programmatically using the `ucimlrepo` package during training. This keeps the repository lightweight while maintaining the original dataset source and attribution.

---

## 🔄 Machine Learning Workflow

```text
                 Dataset
                    │
                    ▼
              Data Loading
                    │
                    ▼
        Data Cleaning & Preprocessing
                    │
                    ▼
             Feature Selection
                    │
                    ▼
             Train/Test Split
                    │
                    ▼
             Model Training
                    │
                    ▼
             Model Evaluation
                    │
                    ▼
              Model Saving
                    │
                    ▼
          Streamlit Prediction App
```

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Used as a baseline binary classification model for predicting whether a machine failure occurs.

### 2. K-Nearest Neighbors (KNN)

Classifies observations based on the similarity between their feature values and neighboring training samples.

### 3. Decision Tree

Uses decision rules based on feature values to classify machine observations into failure and non-failure categories.

---

## 📈 Model Evaluation

The models are evaluated using multiple metrics:

| Metric               | Purpose                                                             |
| -------------------- | ------------------------------------------------------------------- |
| **Accuracy**         | Measures the overall proportion of correct predictions              |
| **Precision**        | Measures how many predicted failures were actually failures         |
| **Recall**           | Measures how many actual failures were correctly identified         |
| **F1-Score**         | Provides a balance between precision and recall                     |
| **Confusion Matrix** | Shows correct and incorrect classification results                  |
| **ROC-AUC**          | Measures the model's ability to distinguish between the two classes |

Accuracy is not considered independently because predictive-maintenance datasets may contain significantly fewer failure cases than normal-operation cases.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **ucimlrepo**
* **Jupyter Notebook**
* **Joblib**

---

## 📁 Project Structure

```text
Industrial-Machine-Failure-Prediction/
│
├── train.py
├── app.py
├── capstone_notebook.ipynb
├── requirements.txt
├── README.md
├── PROJECT_REPORT.pdf
│
├── data/
│   └── Created automatically during execution
│
└── artifacts/
    └── Trained model and evaluation outputs
```

### File Description

| File / Folder             | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `train.py`                | Loads data, preprocesses features, trains models and evaluates performance |
| `app.py`                  | Streamlit application for machine-failure prediction                       |
| `capstone_notebook.ipynb` | Notebook containing analysis and ML workflow                               |
| `requirements.txt`        | Python dependencies required to run the project                            |
| `README.md`               | Project documentation                                                      |
| `PROJECT_REPORT.pdf`      | Technical project report                                                   |
| `data/`                   | Dataset files generated during execution                                   |
| `artifacts/`              | Saved models and evaluation outputs                                        |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Industrial-Machine-Failure-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚂 Training the Model

Run:

```bash
python train.py
```

The training script will:

1. Retrieve the dataset.
2. Load and inspect the data.
3. Perform data cleaning and preproces
