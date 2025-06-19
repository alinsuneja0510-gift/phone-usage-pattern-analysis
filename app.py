import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import silhouette_score

# Setup
st.set_page_config(page_title="📊 Phone Usage Pattern Analysis", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        h1, h2, h3 {color: #0e1117;}
        .stButton>button {background-color: #008080; color: white; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

st.title(":bar_chart: Multi-Page Phone Usage Analyzer")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/phone_usage_india.csv")
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Location'] = le.fit_transform(df['Location'])
    df['Phone Brand'] = le.fit_transform(df['Phone Brand'])
    df['OS'] = le.fit_transform(df['OS'])
    return df

df_raw = load_data()

# Tab Layout
eda_tab, cluster_tab, classify_tab = st.tabs(["\U0001f4c8 EDA", "\U0001f9e0 Clustering", "\U0001f52e Classification"])

# Feature Scaling
df = df_raw.copy()
scaler = StandardScaler()
df_scaled = df.drop(columns=['User ID', 'Primary Use'], errors='ignore')
X_scaled = scaler.fit_transform(df_scaled.select_dtypes(include='number'))

# EDA Tab
with eda_tab:
    st.header(":bar_chart: Exploratory Data Analysis")
    selected_column = st.sidebar.selectbox(":pushpin: Select a Numeric Column", df.select_dtypes(include='number').columns)
    with st.expander(":bar_chart: Feature Distributions", expanded=True):
        st.bar_chart(df[[selected_column]])

    with st.expander(":mag: Feature Correlation", expanded=True):
        st.subheader(":abacus: Feature Correlation Matrix")
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
        st.pyplot(fig)

    with st.expander(":bar_chart: Usage Patterns by Age Group"):
        age_bins = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=["<18", "18-30", "31-45", "46-60", "60+"])
        df['Age Group'] = age_bins
        avg_usage = df.groupby('Age Group')[['Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Social Media Time (hrs/day)', 'Gaming Time (hrs/day)']].mean()
        st.bar_chart(avg_usage)

# Clustering Tab
with cluster_tab:
    st.header(":brain: Clustering Models Comparison")
    models = {
        "KMeans": KMeans(n_clusters=5, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=5),
        "DBSCAN": DBSCAN(eps=2, min_samples=5),
        "Gaussian Mixture": GaussianMixture(n_components=5, random_state=42),
        "Spectral": SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
    }

    for name, model in models.items():
        try:
            if name == "Gaussian Mixture":
                labels = model.fit_predict(X_scaled)
            else:
                labels = model.fit(X_scaled).labels_
            score = silhouette_score(X_scaled, labels)
            st.markdown(f"### ✨ {name} Clustering")
            st.success(f"Silhouette Score: {score:.3f}")

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=50, ax=ax)
            ax.set_title(f"{name} Clustering - PCA Projection")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"{name} failed: {e}")

# Classification Tab
with classify_tab:
    st.header(":crystal_ball: Predict Primary Use")
    model_path = "models/primary_use_classifier.pkl"

    if 'Primary Use' not in df.columns:
        st.error("'Primary Use' column not found in data.")
        st.stop()

    df_clean = df.dropna(subset=['Primary Use'])
    X = df_clean.drop(columns=['User ID', 'Primary Use'], errors='ignore')
    y = df_clean['Primary Use']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader(":robot_face: Choose Model")
    model_name = st.selectbox(":package: Select Model", ["Random Forest", "Logistic Regression", "Decision Tree", "Gradient Boosting"])

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        if model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.2f}")

    fig3, ax3 = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=model.classes_).plot(ax=ax3)
    st.pyplot(fig3)

    st.subheader(":calling: Enter User Details")
    age = st.slider(":birthday: Age", 10, 80, 25)
    gender = st.selectbox(":bust_in_silhouette: Gender", ["Male", "Female", "Other"])
    location = st.slider(":round_pushpin: Location ID", 0, 100, 10)
    phone_brand = st.slider(":iphone: Phone Brand ID", 0, 50, 5)
    os_type = st.selectbox(":computer: Operating System", ["Android", "iOS"])
    screen_time = st.slider(":iphone: Screen Time (hrs/day)", 0.0, 12.0, 4.5)
    data_usage = st.slider(":signal_strength: Data Usage (GB/month)", 0.0, 50.0, 12.0)
    calls_duration = st.slider(":telephone_receiver: Calls Duration (mins/day)", 0, 300, 60)
    apps_installed = st.slider(":calling: Number of Apps Installed", 0, 200, 40)
    social_media = st.slider(":speech_balloon: Social Media Time (hrs/day)", 0.0, 10.0, 2.0)
    ecommerce_spend = st.slider(":shopping_bags: E-commerce Spend (INR/month)", 0, 10000, 1500)
    streaming_time = st.slider(":film_projector: Streaming Time (hrs/day)", 0.0, 10.0, 3.0)
    gaming_time = st.slider(":video_game: Gaming Time (hrs/day)", 0.0, 10.0, 1.0)
    recharge = st.slider(":money_with_wings: Monthly Recharge Cost (INR)", 0, 2000, 300)

    gender_val = 0 if gender == "Male" else 1 if gender == "Female" else 2
    os_val = 0 if os_type == "Android" else 1

    user_input = np.array([[age, gender_val, location, phone_brand, os_val,
                            screen_time, data_usage, calls_duration, apps_installed,
                            social_media, ecommerce_spend, streaming_time,
                            gaming_time, recharge]])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    st.success(f":dart: Predicted Primary Use: **{prediction[0]}**")

    download_df = pd.DataFrame(user_input, columns=X.columns)
    download_df['Predicted Primary Use'] = prediction
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=":arrow_down: Download Prediction Result",
        data=csv,
        file_name="user_primary_use_prediction.csv",
        mime='text/csv'
    )
