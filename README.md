# 📱 Phone Usage Pattern Analysis

An interactive Streamlit dashboard to analyze mobile usage behavior in India, predict primary use types, and cluster users based on device usage patterns.

## 🚀 Features

- EDA Visualizations (screen time, app usage, correlations)
- Predict user's Primary Mobile Use (e.g., Gaming, Education)
- Clustering users by behavior using KMeans
- Data cleaning tool for missing values
- Model comparison tab
- Deployed using Streamlit Cloud

## 🧠 Skills Used

- Python, pandas, NumPy
- Scikit-learn (ML models, clustering)
- Streamlit (dashboard)
- Plotly, Seaborn, Matplotlib
- Joblib (model serialization)

## 📂 Dataset

The dataset includes:
- Demographics (age, gender, location)
- Device specs (brand, OS)
- Usage metrics (screen time, data usage, apps installed)
- Label: Primary Use category

## 📦 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
