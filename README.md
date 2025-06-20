# 📱 Phone Usage Pattern Analysis App

## 📌 Project Overview

This Streamlit web app is an end-to-end data science project focused on analyzing mobile phone usage behavior across users in India. It enables:

1. 📊 Exploratory Data Analysis (EDA) — Visual insights into screen time, app usage, battery usage, etc.  
2. 🎯 Primary Use Classification — Predicting the main mobile usage category (e.g., Gaming, Education).  
3. 🔍 Clustering — Grouping users by behavior using unsupervised learning.  
4. 🧼 Data Cleaning Tool — Upload and fix missing or inconsistent records.  
5. 🧪 Model Comparison — Evaluate performance of multiple classification models.

Deployed using Streamlit Cloud, it provides a fully interactive dashboard.

---

## 📁 Project Structure

```
Project 4/
│
├── app.py                              # Main Streamlit dashboard app
├── requirements.txt                    # Dependencies for deployment
├── README.md                           # This documentation file
│
├── data/                               # Raw input CSV (phone_usage_india.csv)
├── processed_data/                     # Cleaned dataset for EDA and modeling
├── models/
│   ├── classification/                 # Trained classifiers + scaler/encoder
│   └── clustering/                     # Saved clustering models
```

---

## 🧠 Skills Demonstrated

- Python, Pandas, NumPy  
- Scikit-learn (Classification & Clustering)  
- XGBoost, Decision Trees, Random Forests  
- Streamlit Web App Development  
- Plotly, Seaborn (EDA Visualizations)  
- Joblib (Model Saving)  

---

## 🚀 Streamlit App Features

- 📈 EDA dashboards: visualizations for screen time, social media use, battery cost  
- 🧮 Predict the user's Primary Mobile Use: Gaming, Education, Social Media, etc.  
- 🔍 Cluster users with similar behavior patterns  
- 🧼 Upload & clean raw CSV data (missing values, outliers)  
- 🧪 Compare classification model accuracy & predictions  

---

## 🔗 Live App

🌐 [Launch Streamlit App](https://alinsuneja0510-gift.streamlit.app)

---

## 📦 Dataset

The dataset includes:

- Demographics: Age, Gender, Location  
- Device Info: Phone Brand, Operating System  
- Usage Stats:
  - Screen Time (hrs/day)
  - Data Usage (GB/month)
  - App Installs, Call Duration, Recharge Cost
  - App category time (Social Media, Gaming, Streaming, etc.)
- Target: Primary Use (Education, Gaming, Social Media, Entertainment)

File: data/phone_usage_india.csv

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/alinsuneja0510-gift/phone-usage-pattern-analysis.git
cd phone-usage-pattern-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧾 Author

- 👤 Name: Alin Suneja  
- 🔗 GitHub: [alinsuneja0510-gift](https://github.com/alinsuneja0510-gift)  
- 📅 Date: 20 June 2025

---
