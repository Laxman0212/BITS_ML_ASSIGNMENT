import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Wine Quality Classification", layout="wide")
st.title("🍷 Wine Quality Classification Portal")
st.sidebar.info("ML Deployment - 11 Feature Model")

# --------------------------------------------------
# Allowed Features (Order NOT enforced externally)
# --------------------------------------------------
ALLOWED_FEATURES = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Step 1: Upload Test CSV", type="csv")

model_name = st.sidebar.selectbox(
    "Step 2: Select ML Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file:

    try:
        # Load CSV (Wine dataset uses ;)
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("### 📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # Keep original copy for evaluation
        original_df = df.copy()

        # Remove unwanted columns for prediction
        df = df.drop(columns=['quality'], errors='ignore')
        df = df.drop(columns=['target'], errors='ignore')

        uploaded_features = list(df.columns)

        # Validate features
        missing_features = set(ALLOWED_FEATURES) - set(uploaded_features)
        extra_features = set(uploaded_features) - set(ALLOWED_FEATURES)

        if missing_features:
            st.error(f"❌ Missing Features: {missing_features}")
            st.stop()

        if extra_features:
            st.error(f"❌ Extra Features Not Allowed: {extra_features}")
            st.stop()

        # Reorder internally (important for model consistency)
        features = df[ALLOWED_FEATURES]

        # Load model & scaler
        model_file = f"model/{model_name.replace(' ', '_').lower()}.pkl"
        scaler_file = "model/scaler.pkl"

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)

        if st.button("🚀 Run Prediction"):

            # Scale
            features_scaled = scaler.transform(features)

            # Predict
            predictions = model.predict(features_scaled)

            # Add predictions
            original_df["Predicted_Quality"] = predictions

            st.success(f"✅ Predictions completed using {model_name}")
            st.write("### 📊 Prediction Results")
            st.dataframe(original_df)

            # --------------------------------------------------
            # Evaluation Section (if true labels available)
            # --------------------------------------------------
            if 'target' in original_df.columns:

                st.subheader("📈 Performance Analysis")

                y_true = original_df['target']

                # Confusion Matrix
                cm = confusion_matrix(y_true, predictions)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu')
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

                # Classification Report
                st.write("### 📋 Classification Report")
                report = classification_report(y_true, predictions)
                st.text(report)

    except FileNotFoundError:
        st.error("❌ Model or scaler file not found. Run train_models.py first.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.warning("Please upload a CSV containing the 11 required features.")
