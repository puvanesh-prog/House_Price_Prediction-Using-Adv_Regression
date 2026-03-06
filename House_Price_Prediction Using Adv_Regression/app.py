import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet

# --- 1. CONFIGURATION & CLOUD-READY PATHS ---
st.set_page_config(page_title="Real Estate Analytics Hub", page_icon="🏠", layout="wide")

# Path logic: Direct filenames for Cloud deployment
DATA_SOURCE_PATH = "data.csv" 
SAVE_DIR = "house_app_files"
MODEL_PATH = os.path.join(SAVE_DIR, "best_pipeline.joblib")
PROCESSED_DATA_PATH = os.path.join(SAVE_DIR, "house_data_with_predictions.csv")

# --- 2. AUTOMATED MODEL TRAINING WORKFLOW ---
def initialize_model_pipeline():
    # Checking if data.csv exists in the root folder of GitHub repo
    if not os.path.exists(DATA_SOURCE_PATH):
        st.error(f"Critical Error: '{DATA_SOURCE_PATH}' not found in the repository root.")
        st.info("Please ensure your data.csv is uploaded directly to GitHub alongside app.py.")
        st.stop()
    
    with st.spinner("Initializing Predictive Engine on Cloud... Please wait."):
        df_raw = pd.read_csv(DATA_SOURCE_PATH)
        
        # Feature Engineering Logic
        df_raw['TotalSF'] = df_raw['GrLivArea'] + df_raw['TotalBsmtSF']
        df_raw['LogSalePrice'] = np.log1p(df_raw['SalePrice'])
        
        X = df_raw.drop(['SalePrice', 'LogSalePrice'], axis=1)
        y = df_raw['LogSalePrice']
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns
        
        # Preprocessing Pipelines
        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])
        
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))
        ])
        
        # Model Training
        model_pipeline.fit(X, y)
        
        # Generating Predictions
        df_raw['PredictedPrice'] = np.expm1(model_pipeline.predict(X))
        df_raw['PriceDifference_Pct'] = ((df_raw['PredictedPrice'] - df_raw['SalePrice']) / df_raw['SalePrice']) * 100
        
        # Persistent Storage (Automatically creates directory on server)
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        
        joblib.dump(model_pipeline, MODEL_PATH)
        df_raw.to_csv(PROCESSED_DATA_PATH, index=False)
        st.success("System Initialization Complete: Model Serialized on Server.")

# Trigger Training if Assets are Missing (Normal behavior on first Cloud run)
if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSED_DATA_PATH):
    initialize_model_pipeline()

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_application_assets():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model

df, model = load_application_assets()

# --- 4. NAVIGATION INTERFACE ---
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", [
    "🏠 Dashboard", 
    "📊 Exploratory Analysis", 
    "📈 Model Performance", 
    "🔮 Price Predictor", 
    "🔍 SHAP Interpretability", 
    "🏡 Value Recommendations"
])

# --- PAGES 1, 2, 3 & 4 remain the same logic ---
if page == "🏠 Dashboard":
    st.title("🏠 Real Estate Value Analysis System")
    st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80")
    st.markdown("### Executive Summary\nThis platform provides an end-to-end framework for analyzing housing market trends.")

elif page == "📊 Exploratory Analysis":
    st.title("📊 Statistical Data Exploration")
    tab1, tab2 = st.tabs(["Distribution Profiles", "Feature Relationships"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.histogram(df, x="SalePrice", title="Price Distribution"), use_container_width=True)
        with col2: st.plotly_chart(px.histogram(df, x="LogSalePrice", title="Log Distribution"), use_container_width=True)
    with tab2:
        sel = st.selectbox("Select Feature:", df.select_dtypes(include=['object']).columns)
        st.plotly_chart(px.box(df, x=sel, y="SalePrice", title=f"Price by {sel}"), use_container_width=True)

elif page == "📈 Model Performance":
    st.title("📈 Predictive Performance Metrics")
    try:
        st.plotly_chart(px.scatter(df, x='SalePrice', y='PredictedPrice', trendline="ols"), use_container_width=True)
    except:
        st.plotly_chart(px.scatter(df, x='SalePrice', y='PredictedPrice'), use_container_width=True)
    
    st.subheader("Top Influential Features")
    pre = model.named_steps['preprocessor']
    est = model.named_steps['model']
    feat = pd.DataFrame({'Feature': pre.get_feature_names_out(), 'Coef': np.abs(est.coef_)}).sort_values('Coef', ascending=False).head(10)
    st.plotly_chart(px.bar(feat, x='Coef', y='Feature', orientation='h'), use_container_width=True)

elif page == "🔮 Price Predictor":
    st.title("🔮 Real-Time Valuation Engine")
    c1, c2 = st.columns(2)
    with c1:
        qual = st.slider("Quality", 1, 10, 6)
        area = st.number_input("Living Area", value=1800)
        cars = st.slider("Garage", 0, 4, 2)
    with c2:
        bsmt = st.number_input("Basement", value=900)
        year = st.slider("Year", 1950, 2024, 2010)
        nb = st.selectbox("Neighborhood", sorted(df['Neighborhood'].unique()))

    if st.button("Calculate Valuation"):
        cols = df.drop(['SalePrice','PredictedPrice','PriceDifference_Pct','LogSalePrice'], axis=1, errors='ignore').columns
        input_row = pd.DataFrame(columns=cols)
        input_row.loc[0] = 0
        input_row.at[0, 'OverallQual'], input_row.at[0, 'GrLivArea'], input_row.at[0, 'GarageCars'] = qual, area, cars
        input_row.at[0, 'TotalBsmtSF'], input_row.at[0, 'YearBuilt'], input_row.at[0, 'Neighborhood'] = bsmt, year, nb
        pred = np.expm1(model.predict(input_row)[0])
        st.metric("Estimated Market Value", f"${pred:,.2f}")

# --- PAGE 5: SHAP (Optimized for Cloud) ---
elif page == "🔍 SHAP Interpretability":
    st.title("🔍 Model Explainability (SHAP)")
    st.info("Processing SHAP values on server...")
    
    cols = df.drop(['SalePrice','PredictedPrice','PriceDifference_Pct','LogSalePrice'], axis=1, errors='ignore').columns
    X_sample = df.drop(['SalePrice','PredictedPrice','PriceDifference_Pct','LogSalePrice'], axis=1, errors='ignore').sample(min(50, len(df)))
    X_trans = model.named_steps['preprocessor'].transform(X_sample)
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    
    explainer = shap.Explainer(model.named_steps['model'], X_trans)
    shap_vals = explainer(X_trans)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_trans, feature_names=model.named_steps['preprocessor'].get_feature_names_out(), show=False)
    plt.tight_layout()
    st.pyplot(fig)

# --- PAGE 6: RECOMMENDATIONS ---
elif page == "🏡 Value Recommendations":
    st.title("🏡 Strategic Investment Insights")
    max_b = st.sidebar.slider("Budget", 50000, 750000, 400000)
    min_q = st.sidebar.slider("Min Quality", 1, 10, 7)
    
    recom = df[(df['SalePrice'] <= max_b) & (df['OverallQual'] >= min_q)].copy()
    if not recom.empty:
        recom = recom.sort_values('PriceDifference_Pct', ascending=False).head(10)
        st.dataframe(recom[['Neighborhood', 'OverallQual', 'SalePrice', 'PredictedPrice', 'PriceDifference_Pct']].style.format({
            'SalePrice': '${:,.2f}', 'PredictedPrice': '${:,.2f}', 'PriceDifference_Pct': '{:.2f}%'
        }))
    else:
        st.warning("No matches found.")
