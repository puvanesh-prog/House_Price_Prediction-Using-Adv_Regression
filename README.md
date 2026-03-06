# 🏠 Real Estate Value Analysis & Prediction Hub

An interactive, data-driven web application that leverages **Advanced Regression Techniques** to predict property valuations and identify high-value investment opportunities.

## 🚀 Live Demo
**Check out the live app here:** https://bit.ly/4udFFti 

---

## 📌 Project Overview
This project aims to solve the transparency issue in real estate pricing. By using an **ElasticNet Regression** model, it provides accurate market valuations based on physical attributes and location.

### 🔑 Key Modules:
- **📈 Dashboard**: Executive summary of market trends.
- **📊 Statistical EDA**: Deep-dive into price distributions and categorical variances.
- **🔮 Valuation Engine**: Real-time price prediction based on user inputs (Square Feet, Quality, Neighborhood).
- **🔍 Model Explainability**: Utilizing **SHAP values** to visualize which features (like Living Area or Quality) impact the price the most.
- **🏡 Strategic Recommendations**: A custom algorithm that identifies the **Top 10 Undervalued Assets** based on the gap between market price and predicted value.

---

## 🛠️ Tech Stack
- **Language**: Python
- **Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (ElasticNet, Pipelines, ColumnTransformers)
- **Interpretability**: SHAP (Shapley Additive Explanations)
- **Visuals**: Plotly Express, Matplotlib
- **Deployment**: Streamlit Community Cloud

---

## 📂 Repository Structure
```text
├── app.py                # Main Streamlit application logic
├── data.csv              # Raw housing dataset
├── requirements.txt      # Project dependencies (Streamlit, Scikit-learn, etc.)
└── house_app_files/      # (Auto-generated) Serialized model and processed data
