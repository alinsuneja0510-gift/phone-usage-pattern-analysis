import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

st.set_page_config(page_title="📱 Mobile Usage Dashboard", layout="wide", initial_sidebar_state="expanded")

# Tabs
tabs = st.tabs([
    "📊 EDA",
    "📈 Classification",
    "🔍 Clustering",
    "🧼 Data Cleaner",
    "🧪 Model Comparison",
    "🚀 Deployment Guide"
])

@st.cache_data
def load_data():
    df = pd.read_csv("processed_data/eda_data.csv")
    return df

# --- EDA ---
with tabs[0]:
    st.title("📊 Exploratory Data Analysis")
    eda_df = load_data()

    st.subheader("📌 Distribution of App Usage Time by Primary Use")
    usage_time_cols = ["Social Media Time (hrs/day)", "Streaming Time (hrs/day)", "Gaming Time (hrs/day)"]
    selected_usage = st.selectbox("Select a feature:", usage_time_cols)
    if selected_usage in eda_df.columns:
        fig1 = px.histogram(eda_df, x=selected_usage, color="Primary Use", barmode="overlay", nbins=40)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.error(f"'{selected_usage}' not found in uploaded dataset")

    st.subheader("🔋 Screen-on Time vs Battery Consumption")
    if "Screen Time (hrs/day)" in eda_df.columns and "Monthly Recharge Cost (INR)" in eda_df.columns:
        fig2 = px.scatter(
            eda_df,
            x="Screen Time (hrs/day)",
            y="Monthly Recharge Cost (INR)",
            color="Primary Use",
            size="Data Usage (GB/month)",
            title="Screen Time vs Battery Cost by Primary Use"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📶 Data Usage vs App Installations by Age")
    fig3 = px.scatter(
        eda_df,
        x="Age",
        y="Data Usage (GB/month)",
        size="Number of Apps Installed",
        color="Primary Use",
        hover_data=['Gender']
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🌡️ Feature Correlation Heatmap")
    numeric_data = eda_df.select_dtypes(include=np.number)
    fig4 = px.imshow(numeric_data.corr(), text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🧪 Screen Time Boxplot by User Class")
    fig5 = px.box(eda_df, x="Primary Use", y="Screen Time (hrs/day)", color="Primary Use")
    st.plotly_chart(fig5, use_container_width=True)

# --- Classification Tab ---
with tabs[1]:
    st.title("📈 Primary Use Prediction")
    st.info("""
    This section lets you enter key mobile usage statistics and predicts the user's **Primary Mobile Usage**.
    👉 Fill in the inputs below and hit **Predict** to see the AI's guess!
    """)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    location = st.selectbox("Location", ["Mumbai", "Delhi", "Ahmedabad", "Pune"])
    phone_brand = st.selectbox("Phone Brand", ["Vivo", "Realme", "Nokia", "Samsung", "Xiaomi"])
    os_type = st.selectbox("Operating System", ["Android", "iOS"])

    age = st.slider("Age", 10, 80, 30)
    screen_time = st.slider("Screen Time (hrs/day)", 0.0, 16.0, 5.0)
    data_usage = st.slider("Data Usage (GB/month)", 0.0, 50.0, 10.0)
    calls = st.slider("Calls Duration (mins/day)", 0.0, 300.0, 60.0)
    apps = st.slider("Number of Apps Installed", 0, 300, 100)
    social_time = st.slider("Social Media Time (hrs/day)", 0.0, 10.0, 3.0)
    ecommerce = st.slider("E-commerce Spend (INR/month)", 0, 10000, 2000)
    streaming = st.slider("Streaming Time (hrs/day)", 0.0, 10.0, 2.0)
    gaming = st.slider("Gaming Time (hrs/day)", 0.0, 10.0, 2.0)
    recharge = st.slider("Monthly Recharge Cost (INR)", 0, 3000, 1000)

    if st.button("Predict"):
        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        os_map = {"Android": 0, "iOS": 1}
        loc_map = {"Mumbai": 2, "Delhi": 0, "Ahmedabad": 1, "Pune": 3}
        brand_map = {"Vivo": 4, "Realme": 3, "Nokia": 2, "Samsung": 1, "Xiaomi": 0}

        features = [[
            age,
            gender_map[gender],
            loc_map[location],
            brand_map[phone_brand],
            os_map[os_type],
            screen_time,
            data_usage,
            calls,
            apps,
            social_time,
            ecommerce,
            streaming,
            gaming,
            recharge
        ]]

        scaler = joblib.load("models/classification/scaler.pkl")
        model = joblib.load("models/classification/random_forest.pkl")
        target_encoder = joblib.load("models/classification/target_encoder.pkl")

        features_scaled = scaler.transform(features)
        probs = model.predict_proba(features_scaled)[0]
        pred = model.predict(features_scaled)
        label = target_encoder.inverse_transform(pred)[0]

        emoji_map = {
            "Gaming": "🎮",
            "Social Media": "💬",
            "Entertainment": "🎬",
            "Education": "📚",
            "Work": "💼"
        }

        st.success(f"Predicted Primary Use: **{emoji_map.get(label, '📱')} {label}**")

        st.markdown("### 🔢 Prediction Probabilities")
        proba_df = pd.DataFrame({
            "Category": target_encoder.inverse_transform(np.arange(len(probs))),
            "Probability": np.round(probs * 100, 2)
        }).sort_values("Probability", ascending=False)
        st.table(proba_df)

# --- Clustering Tab ---
with tabs[2]:
    st.title("🔍 Cluster User Behavior")
    st.info("Upload data and apply clustering algorithms.")
    uploaded_cluster = st.file_uploader("Upload dataset for clustering", type="csv")
    if uploaded_cluster:
        df_cluster = pd.read_csv(uploaded_cluster)
        st.dataframe(df_cluster.head())

        num_clusters = st.slider("Select number of clusters", 2, 10, 3)

        st.subheader("Select features for clustering")
        features = st.multiselect("Choose numeric features", df_cluster.select_dtypes(include=np.number).columns.tolist())

        if st.button("Apply Clustering") and features:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[features])
            st.success("Clustering applied!")
            fig = px.scatter_matrix(df_cluster, dimensions=features, color='Cluster')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_cluster.head())

# --- Data Cleaning Tab ---
with tabs[3]:
    st.title("🧼 Data Cleaning Assistant")
    st.info("This tool helps you inspect and fix missing or inconsistent data.")
    uploaded_clean = st.file_uploader("Upload raw dataset for cleaning", type="csv")
    if uploaded_clean:
        raw_df = pd.read_csv(uploaded_clean)
        st.dataframe(raw_df.head())

        st.subheader("🧹 Missing Values Summary")
        missing = raw_df.isnull().sum()
        st.write(missing[missing > 0])

        if st.button("Drop Rows with Missing Values"):
            cleaned_df = raw_df.dropna()
            st.success("Rows with missing values dropped.")
            st.dataframe(cleaned_df.head())

# --- Model Comparison Tab ---
with tabs[4]:
    st.title("🧪 Model Comparison")
    st.info("View and compare the performance of different classification models.")

    models = ["logistic_regression", "decision_tree", "random_forest", "svm", "xgboost"]
    report_df = []
    y_test = joblib.load("models/classification/y_test.pkl")

    for model_name in models:
        model = joblib.load(f"models/classification/{model_name}.pkl")
        y_pred = model.predict(joblib.load("models/classification/X_test.pkl"))
        report = classification_report(y_test, y_pred, output_dict=True)
        acc = report['accuracy']
        report_df.append({"Model": model_name, "Accuracy": acc})

    st.table(pd.DataFrame(report_df).sort_values("Accuracy", ascending=False))

# --- Deployment Tab ---
with tabs[5]:
    st.title("🚀 Deployment Guide")
    st.markdown("""
        You can deploy this app using:

        **✅ Streamlit Cloud:**
        - Push your code to GitHub
        - Log into [Streamlit Cloud](https://streamlit.io/cloud)
        - Click **New App** and select the repo

        **🖥️ Local Deployment:**
        ```bash
        streamlit run app.py
        ```
    """)
