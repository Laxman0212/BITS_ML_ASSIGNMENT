# 1. Install required libraries (if running in lab / notebook)
# !pip install xgboost scikit-learn pandas matplotlib seaborn
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# --------------------------------------------------
# 2. Create model directory
# --------------------------------------------------
os.makedirs("model", exist_ok=True)

# --------------------------------------------------
# 3. Load Dataset (UCI Wine Quality - Red Wine)
# --------------------------------------------------
df = pd.read_csv("winequality-red.csv", sep=';')

# Create binary target
# 1 = Good (quality >= 6), 0 = Bad
df['target'] = (df['quality'] >= 6).astype(int)

X = df.drop(['quality', 'target'], axis=1)
y = df['target']

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 5. Scaling (Important for Streamlit compatibility)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Save Scaler for Streamlit
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully.")

# --------------------------------------------------
# 6. Define Models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# --------------------------------------------------
# 7. Train, Evaluate & Save Models
# --------------------------------------------------
results = []

for name, model in models.items():

    # Train on scaled data (consistent with Streamlit)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Save model inside /model folder
    model_filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    print(f"{name} saved successfully.")

    # Store evaluation metrics
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# --------------------------------------------------
# 8. Display Model Comparison
# --------------------------------------------------
comparison_df = pd.DataFrame(results)

print("\nModel Comparison Results:\n")
print(comparison_df.to_markdown(index=False))
