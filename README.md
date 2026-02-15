# 🍷 Wine Quality Classification Portal

### BITS Machine Learning Assignment 2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green.svg)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen.svg)

------------------------------------------------------------------------

## 📌 Problem Statement

The goal of this project is to classify **red wine quality** as  
**"Good"** or **"Bad"** based on its physicochemical properties.

This is a **binary classification task** where:

- Wine is considered **Good** if quality score ≥ 6  
- Wine is considered **Bad** if quality score < 6  

------------------------------------------------------------------------

## 📊 Dataset Description

- **Source:** UCI Machine Learning Repository (Wine Quality Dataset – Red)
- **Total Instances:** 1,599
- **Input Features:** 11

### Feature List:

- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

### Target Variable:

Binary Target (`target`):

- 0 → Bad Wine  
- 1 → Good Wine  

------------------------------------------------------------------------

## 🤖 Machine Learning Models Used

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.7406   | 0.8191  | 0.7857    | 0.7374  | 0.7608   | 0.4793  |
| Decision Tree        | 0.7344   | 0.7325  | 0.7701    | 0.7486  | 0.7592   | 0.4634  |
| kNN                  | 0.7063   | 0.7737  | 0.7202    | 0.7765  | 0.7473   | 0.3994  |
| Naive Bayes          | 0.7344   | 0.7927  | 0.7582    | 0.7710  | 0.7645   | 0.4600  |
| Random Forest        | 0.7844   | 0.8919  | 0.8056    | 0.8101  | 0.8078   | 0.5623  |
| XGBoost              | 0.8125   | 0.8787  | 0.8362    | 0.8268  | 0.8315   | 0.6203  |

------------------------------------------------------------------------

## 📈 Observations on Model Performance

| Model Name           | Observation |
|----------------------|-------------|
| Logistic Regression  | Strong baseline model with good AUC and balanced precision-recall after feature scaling. |
| Decision Tree        | Captured non-linear patterns but slightly lower AUC compared to ensemble methods. |
| kNN                  | Performance influenced by scaling and neighborhood size; moderate results overall. |
| Naive Bayes          | Computationally efficient but assumes feature independence, limiting performance slightly. |
| Random Forest        | Significant improvement in Accuracy and AUC due to ensemble averaging and reduced variance. |
| XGBoost              | Best performing model with highest Accuracy, F1 Score, and MCC due to boosting mechanism. |

------------------------------------------------------------------------

## 🚀 Streamlit Application

This project includes a **Wine Quality Prediction Portal** built using **Streamlit**.

### Features of the Web App:

- Upload CSV file containing the 11 required features
- Select any of the 6 trained models
- Automatic feature validation (only 11 allowed features)
- Internal feature reordering (order not required)
- Standardization using saved scaler
- Generate predictions
- Display Confusion Matrix (if target column provided)
- Display Classification Report

------------------------------------------------------------------------

## ⚙️ Installation & Setup

Install dependencies:

```bash
pip install -r requirements.txt

Run the Streamlit App:

```bash
python -m streamlit run app.py
```

------------------------------------------------------------------------

## 📁 Project Structure

    project/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── train_models.py
    └── model/
        ├── logistic_regression.pkl
        ├── decision_tree.pkl
        ├── knn.pkl
        ├── naive_bayes.pkl
        ├── random_forest.pkl
        └── xgboost.pkl

------------------------------------------------------------------------

## 📌 Author

**Laxman**\
Embedded Engineer \| ML Enthusiast
