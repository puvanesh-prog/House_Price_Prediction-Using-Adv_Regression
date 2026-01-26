import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("🔍 SHAP Model Explainability")

csv_path = "house_app_files/house_data_with_predictions.csv"
pipeline_path = "house_app_files/best_pipeline.pkl"

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(csv_path)
st.success("✅ Data Loaded Successfully!")

try:
    with open(pipeline_path, "rb") as f:
        pipeline = cloudpickle.load(f)
    st.info("✅ Loaded saved pipeline")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']  # Renamed back to 'model'

num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

X = df[num_cols.tolist() + cat_cols.tolist()]
X_sample = X.sample(min(200, len(X)), random_state=42)
X_sample_transformed = to_dense(preprocessor.transform(X_sample))

ohe = preprocessor.named_transformers_['cat']
feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))

# Compute SHAP values without hashing the model
@st.cache_data
def compute_shap(_model, X_transformed):
    explainer = shap.Explainer(_model, X_transformed)
    return explainer(X_transformed)

shap_values = compute_shap(model, X_sample_transformed)

st.subheader("SHAP Summary Plot (Global Feature Importance)")
fig_summary = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample_transformed, feature_names=feature_names, show=False)
st.pyplot(fig_summary)

st.subheader("SHAP Waterfall Plot (Single Prediction Explanation)")
selected_idx = st.number_input(
    "Select a row index to explain:",
    min_value=0,
    max_value=len(X) - 1,
    value=0
)

single_row = X.iloc[[selected_idx]]
single_transformed = to_dense(preprocessor.transform(single_row))
shap_single = shap.Explainer(model, X_sample_transformed)(single_transformed)

fig_waterfall = plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single[0], show=False)
st.pyplot(fig_waterfall)

st.subheader(" Original Data and Predicted Price")
st.write(single_row)

pred_log = model.predict(single_transformed)[0]
pred_price = np.expm1(pred_log)
st.success(f"Predicted Sale Price: ${pred_price:,.0f}")
