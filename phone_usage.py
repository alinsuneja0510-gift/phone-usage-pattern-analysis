import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\Admin\Project 4\data\phone_usage_india.csv")

# Drop User ID column if it exists
if 'User ID' in df.columns:
    df.drop(columns=['User ID'], inplace=True)

# 2. Define column types
target = "Primary Use"
cat_cols = ['Gender', 'Location', 'Phone Brand', 'OS']
num_cols = df.drop(columns=cat_cols + [target]).columns.tolist()

# 3. Impute missing numeric values
imputer = SimpleImputer(strategy="mean")
df[num_cols] = imputer.fit_transform(df[num_cols])

# 4. Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Encode target
target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])
# Save target encoder for Streamlit
joblib.dump(target_encoder, "models/classification/target_encoder.pkl")

# 6. Split features and target
X = df.drop(columns=[target])
y = df[target]

# 7. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save EDA-ready version
eda_df = df.copy()
eda_df[target] = target_encoder.inverse_transform(eda_df[target])
eda_df.to_csv("processed_data/eda_data.csv", index=False)

# Save processed data
os.makedirs("processed_data", exist_ok=True)
pd.DataFrame(X_scaled).to_csv("processed_data/cleaned_data.csv", index=False)

joblib.dump(scaler, "models/classification/scaler.pkl")
joblib.dump(label_encoders, "models/classification/label_encoders.pkl")

# 8. Train Classification Models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ ADD THESE LINES:
joblib.dump(X_test, "models/classification/X_test.pkl")
joblib.dump(y_test, "models/classification/y_test.pkl")

classifiers = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(),
    "xgboost": XGBClassifier(eval_metric='mlogloss')

}

os.makedirs("models/classification", exist_ok=True)

for name, model in classifiers.items():
    print(f"\n🔍 Training classifier: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"models/classification/{name}.pkl")

# 9. Train Clustering Models
cluster_models = {
    "KMeans": KMeans(n_clusters=5, n_init='auto', random_state=42),
    "dbscan": DBSCAN(),
    "agglomerative": AgglomerativeClustering(n_clusters=5),
    "gmm": GaussianMixture(n_components=5, random_state=42),
    "birch": Birch(n_clusters=5)
}

os.makedirs("models/clustering", exist_ok=True)

import cloudpickle

for name, model in cluster_models.items():
    print(f"\n🔍 Training clusterer: {name}")
    labels = model.fit_predict(X_scaled)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster Distribution: {dict(zip(unique, counts))}")
    
    # Save using cloudpickle
    with open(f"models/clustering/{name}.pkl", "wb") as f:
        cloudpickle.dump(model, f)
