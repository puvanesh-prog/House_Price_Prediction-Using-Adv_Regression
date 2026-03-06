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

# --- 1. CONFIGURATION & PATHS ---
st.set_page_config(page_title="Real Estate Analytics Hub", page_icon="🏠", layout="wide")

# Correct Windows Path
DATA_SOURCE_PATH = "data.csv"
SAVE_DIR = "house_app_files"
MODEL_PATH = os.path.join(SAVE_DIR, "best_pipeline.joblib")
PROCESSED_DATA_PATH = os.path.join(SAVE_DIR, "house_data_with_predictions.csv")

# --- 2. AUTOMATED MODEL TRAINING WORKFLOW ---
def initialize_model_pipeline():
    if not os.path.exists(DATA_SOURCE_PATH):
        st.error(f"Critical Error: Source data not found at {DATA_SOURCE_PATH}")
        st.stop()
    
    with st.spinner("Initializing Predictive Engine... Please wait."):
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
        
        # Consistent Column Naming: PriceDifference_Pct
        df_raw['PredictedPrice'] = np.expm1(model_pipeline.predict(X))
        df_raw['PriceDifference_Pct'] = ((df_raw['PredictedPrice'] - df_raw['SalePrice']) / df_raw['SalePrice']) * 100
        
        # Persistent Storage
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        joblib.dump(model_pipeline, MODEL_PATH)
        df_raw.to_csv(PROCESSED_DATA_PATH, index=False)
        st.success("System Initialization Complete.")

# Check if training is needed
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

# --- PAGE 1: DASHBOARD ---
if page == "🏠 Dashboard":
    st.title("🏠 Real Estate Value Analysis System")
    st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80")
    st.markdown("""
    ### Executive Summary
    This platform provides an end-to-end framework for analyzing housing market trends and predicting property valuations.
    
    **Analytical Modules:**
    - **EDA**: Statistical visualization of market variables.
    - **Model Performance**: Accuracy and feature weight insights.
    - **Valuation Engine**: Real-time property price estimation.
    - **Investment Recommendations**: Identification of undervalued assets.
    """)

# --- PAGE 2: EDA ---
elif page == "📊 Exploratory Analysis":
    st.title("📊 Statistical Data Exploration")
    tab1, tab2 = st.tabs(["Distribution Profiles", "Feature Relationships"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df, x="SalePrice", title="Primary Price Distribution", color_discrete_sequence=['#2E86C1']), use_container_width=True)
        with col2:
            st.plotly_chart(px.histogram(df, x="LogSalePrice", title="Log-Normalized Distribution", color_discrete_sequence=['#28B463']), use_container_width=True)
    with tab2:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        sel = st.selectbox("Select Categorical Feature:", cat_cols)
        st.plotly_chart(px.box(df, x=sel, y="SalePrice", title=f"Valuation Variance by {sel}", color=sel), use_container_width=True)

# --- PAGE 3: MODEL PERFORMANCE ---
elif page == "📈 Model Performance":
    st.title("📈 Predictive Performance Metrics")
    # statsmodels should be installed for trendline
    try:
        st.plotly_chart(px.scatter(df, x='SalePrice', y='PredictedPrice', trendline="ols", title="Actual vs. Predicted Valuation"), use_container_width=True)
    except:
        st.plotly_chart(px.scatter(df, x='SalePrice', y='PredictedPrice', title="Actual vs. Predicted Valuation"), use_container_width=True)
        st.warning("Install 'statsmodels' to see the regression trendline.")
    
    st.subheader("Top Influential Features (ElasticNet Coefficients)")
    preprocessor = model.named_steps['preprocessor']
    est = model.named_steps['model']
    feat_names = preprocessor.get_feature_names_out()
    coefs = pd.DataFrame({'Feature': feat_names, 'AbsoluteCoefficient': np.abs(est.coef_)}).sort_values('AbsoluteCoefficient', ascending=False).head(10)
    st.plotly_chart(px.bar(coefs, x='AbsoluteCoefficient', y='Feature', orientation='h', color='AbsoluteCoefficient'), use_container_width=True)

# --- PAGE 4: PREDICTOR ---
elif page == "🔮 Price Predictor":
    st.title("🔮 Real-Time Valuation Engine")
    c1, c2 = st.columns(2)
    with c1:
        qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        area = st.number_input("Living Area (Sq Ft)", value=1800)
        cars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
    with c2:
        bsmt = st.number_input("Total Basement (Sq Ft)", value=900)
        year = st.slider("Year Constructed", 1950, 2024, 2010)
        nb = st.selectbox("Neighborhood Location", sorted(df['Neighborhood'].unique()))

    if st.button("Calculate Valuation"):
        # Match training features
        cols_to_drop = ['SalePrice','PredictedPrice','PriceDifference_Pct','LogSalePrice']
        input_row = pd.DataFrame(columns=df.drop(cols_to_drop, axis=1, errors='ignore').columns)
        input_row.loc[0] = 0
        input_row.at[0, 'OverallQual'] = qual
        input_row.at[0, 'GrLivArea'] = area
        input_row.at[0, 'GarageCars'] = cars
        input_row.at[0, 'TotalBsmtSF'] = bsmt
        input_row.at[0, 'YearBuilt'] = year
        input_row.at[0, 'Neighborhood'] = nb
        
        pred = np.expm1(model.predict(input_row)[0])
        st.metric("Estimated Market Value", f"${pred:,.2f}")

# --- PAGE 5: SHAP ---
elif page == "🔍 SHAP Interpretability":
    st.title("🔍 Model Explainability (SHAP Analysis)")
    st.info("Generating SHAP summary plot...")
    
    cols_to_drop = ['SalePrice','PredictedPrice','PriceDifference_Pct','LogSalePrice']
    X_sample = df.drop(cols_to_drop, axis=1, errors='ignore').sample(min(50, len(df)))
    X_trans = model.named_steps['preprocessor'].transform(X_sample)
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    
    explainer = shap.Explainer(model.named_steps['model'], X_trans)
    shap_vals = explainer(X_trans)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X_trans, feature_names=model.named_steps['preprocessor'].get_feature_names_out(), show=False)
    st.pyplot(plt.gcf())

# --- PAGE 6: RECOMMENDATIONS (FIXED VERSION) ---
elif page == "🏡 Value Recommendations":
    st.title("🏡 Strategic Investment Insights")
    
    # Selection criteria from sidebar
    max_budget = st.sidebar.slider("Investment Budget", 50000, 750000, 400000)
    min_quality = st.sidebar.slider("Minimum Quality Threshold", 1, 10, 7)
    
    # Filter based on existing columns in DF
    # We use 'PriceDifference_Pct' consistently here
    recommendations = df[(df['SalePrice'] <= max_budget) & (df['OverallQual'] >= min_quality)].copy()
    
    if not recommendations.empty:
        # Sort by best value (highest difference between Predicted and Actual)
        recommendations = recommendations.sort_values('PriceDifference_Pct', ascending=False).head(10)
        
        st.subheader(f"Top 10 Undervalued Assets (Budget: < ${max_budget:,})")
        
        # Display clean dataframe
        display_cols = ['Neighborhood', 'OverallQual', 'SalePrice', 'PredictedPrice', 'PriceDifference_Pct']
        st.dataframe(recommendations[display_cols].style.format({
            'SalePrice': '${:,.2f}',
            'PredictedPrice': '${:,.2f}',
            'PriceDifference_Pct': '{:.2f}%'
        }))
        
        st.success("Tip: Properties with high positive 'PriceDifference_Pct' are estimated to be worth more than their sale price.")
    else:
        st.warning("No properties match your current filters. Try increasing the budget or lowering the quality threshold.")