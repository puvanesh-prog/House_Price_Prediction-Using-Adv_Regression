import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os

st.set_page_config(page_title="Predict House Price", layout="wide")
st.title("🔮 Predict House Price")

csv_path = "house_app_files/house_data_with_predictions.csv"
df = pd.read_csv(csv_path)

pipeline_path = "house_app_files/best_pipeline.pkl"
if os.path.exists(pipeline_path):
    with open(pipeline_path, "rb") as f:
        model = cloudpickle.load(f)
    st.info("✅ Loaded saved pipeline")
else:
    st.error("❌ Pipeline not found!")
    st.stop()

numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['Neighborhood']

st.sidebar.header("Set Feature Values")
input_data = {}
for f in numeric_features:
    min_val, max_val = int(df[f].min()), int(df[f].max())
    default_val = int(df[f].median())
    step = 1 if max_val < 50 else 50 if max_val > 1000 else 10
    input_data[f] = st.sidebar.slider(f, min_val, max_val, default_val, step=step)

input_data['Neighborhood'] = st.sidebar.selectbox(
    "Neighborhood", sorted(df['Neighborhood'].unique())
)

input_df = pd.DataFrame([input_data])

all_cols = df.drop(['SalePrice','LogSalePrice'], axis=1).columns
for col in all_cols:
    if col not in input_df.columns:
        input_df[col] = 0 if df[col].dtype in [np.int64, np.float64] else 'None'

input_df = input_df[all_cols]

pred_price = np.expm1(model.predict(input_df)[0])

st.subheader("Predicted House Price")
st.success(f"${pred_price:,.0f}")
st.write("### Input Features")
st.table(input_df[numeric_features + categorical_features])
