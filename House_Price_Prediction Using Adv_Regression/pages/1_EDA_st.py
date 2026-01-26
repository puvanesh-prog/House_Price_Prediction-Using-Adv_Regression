import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="EDA - Housing Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis (EDA)")

@st.cache_data
def load_data():
    return pd.read_csv("house_app_files/house_data_with_predictions.csv")

df = load_data()

st.success("Data Loaded Successfully!")


st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# SalePrice Distribution
st.subheader("House Sale Price Distribution")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="SalePrice", nbins=50,
                       title="SalePrice Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    df_log = df.copy()
    df_log["LogSalePrice"] = np.log1p(df["SalePrice"])
    fig_log = px.histogram(df_log, x="LogSalePrice", nbins=50,
                           title="Log-Transformed SalePrice")
    st.plotly_chart(fig_log, use_container_width=True)

# Correlation Section
st.subheader("🔗 Feature Correlations with SalePrice")

# Buttons
option = st.radio(
    "Select Correlation Type:",
    ["Normal Correlation", "Log-Transformed Correlation"]
)

numeric_df = df.select_dtypes(include=["int64", "float64"])

if option == "Normal Correlation":
    corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)[1:11]
    corr_df = corr.reset_index().rename(
        columns={'index': 'Feature', 'SalePrice': 'Correlation'}
    )
    title = "Top 10 Correlated Features (SalePrice)"

else:
    df_log = df.copy()
    df_log["LogSalePrice"] = np.log1p(df_log["SalePrice"])
    numeric_log_df = df_log.select_dtypes(include=["int64", "float64"])
    corr = numeric_log_df.corr()['LogSalePrice'].sort_values(ascending=False)[1:10]
    corr_df = corr.reset_index().rename(
        columns={'index': 'Feature', 'LogSalePrice': 'Correlation'}
    )
    title = "Top 10 Correlated Features (LogSalePrice)"

# Plot correlation
fig_corr = px.bar(
    corr_df,
    x="Feature",
    y="Correlation",
    title=title,
    color="Correlation"
)
st.plotly_chart(fig_corr, use_container_width=True)

# Categorical Analysis
st.subheader("Categorical Feature Analysis")

cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(cat_cols) > 0:
    selected_cat = st.selectbox("Select a Categorical Column:", cat_cols)
    
    fig_cat = px.box(df, x=selected_cat, y="SalePrice",
                     title=f"SalePrice vs {selected_cat}")
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("No categorical columns found.")

st.info("✔ EDA page loaded successfully.")
