import streamlit as st

st.set_page_config(
    page_title="House Price App",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Analysis & Recommendations")

st.markdown("""
Use the sidebar to navigate between pages:

- **EDA**: Explore data distributions and log-transformed analysis  
- **Model**: View model insights and feature importance  
- **Predict**: Predict house prices interactively  
- **Recommendation**: Get top house deals based on filters
""")
