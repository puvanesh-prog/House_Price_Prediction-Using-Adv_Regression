import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="wide")
st.title("📈 Model Insights")

csv_path = "house_app_files/house_data_with_predictions.csv"
pipeline_path = "house_app_files/best_pipeline.joblib"  

df = pd.read_csv(csv_path)
best_enet = joblib.load(pipeline_path)

# Actual vs Predicted Prices 
st.subheader("Actual vs Predicted Prices")
fig = px.scatter(
    df,
    x='SalePrice',
    y='PredictedPrice',
    hover_data=['Neighborhood', 'GrLivArea'],
    title="Actual vs Predicted Prices"
)
fig.add_shape(
    type="line",
    line=dict(dash="dash", color="red"),
    x0=df['SalePrice'].min(), x1=df['SalePrice'].max(),
    y0=df['SalePrice'].min(), y1=df['SalePrice'].max()
)
st.plotly_chart(fig, use_container_width=True)

# Top 10 Features (ElasticNet Coefficients) ---
st.subheader("Top 10 Influential Features")
preprocessor = best_enet.named_steps['preprocessor']
model = best_enet.named_steps['model']

try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

feature_names = [f.split("__")[-1] if "__" in f else f for f in feature_names]

coefs = model.coef_ if hasattr(model, 'coef_') else np.zeros(len(feature_names))
feat_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs, 'AbsCoeff': np.abs(coefs)})
top10 = feat_imp.sort_values('AbsCoeff', ascending=False).head(10)

fig2 = px.bar(top10, x='AbsCoeff', y='Feature', orientation='h', title="Top 10 Features - ElasticNet", color='AbsCoeff')
fig2.update_yaxes(categoryorder='total ascending')
st.plotly_chart(fig2, use_container_width=True)
