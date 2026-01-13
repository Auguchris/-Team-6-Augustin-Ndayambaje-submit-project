# =========================================================
# STEP 1: Import Libraries
# =========================================================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# STEP 2: Read Dataset
# =========================================================
data_path = r"C:\Users\auguc\Documents\GitHub\project DSCI  COHORT1\DATA\Telecom_Service_Quality_Rwanda_2023_2025.xlsx"

df = pd.read_excel(data_path)

# Standardize column names (CRITICAL FIX)
df.columns = df.columns.str.strip().str.lower()

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# =========================================================
# STEP 3: Encode Satisfaction Label (TARGET VARIABLE)
# =========================================================
mapping = {
    "Dissatisfied": 0,
    "Neutral": 1,
    "Satisfied": 2
}

df["satisfaction_label_encoded"] = df["satisfaction_label"].map(mapping)

# Check encoding
print("\nLabel Encoding Check:")
print(df[["satisfaction_label", "satisfaction_label_encoded"]].head())

# =========================================================
# STEP 4: Feature Scaling (Numerical QoS Indicators)
# =========================================================
num_cols = [
    "downlink_throughput_kbps",
    "uplink_throughput_kbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "rsrp_dbm",
    "rsrq_db",
    "cell_load_pct",
    "active_users",
    "cssr_pct",
    "dcr_pct",
    "mos_voice",
    "customer_qoe_score"
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save cleaned dataset
df.to_excel(
    "DATA/Telecom_Service_Quality_Rwanda_2023_2025_cleaned.xlsx",
    index=False
)

# =========================================================
# STEP 5: Exploratory Data Analysis (EDA)
# =========================================================

# 5.1 Distribution of Latency
plt.figure(figsize=(6,4))
plt.hist(df["latency_ms"], bins=30)
plt.xlabel("Latency (Standardized)")
plt.ylabel("Frequency")
plt.title("Distribution of Network Latency")
plt.show()

# 5.2 Correlation Analysis
corr_matrix = df[num_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
plt.title("Correlation Matrix of QoS Indicators")
plt.show()

# =========================================================
# STEP 6: Satisfaction vs QoS Analysis
# =========================================================
latency_by_sat = df.groupby("satisfaction_label")["latency_ms"].mean()
throughput_by_sat = df.groupby("satisfaction_label")[
    "downlink_throughput_kbps"
].mean()

print("\nAverage Latency by Satisfaction:\n", latency_by_sat)
print("\nAverage Downlink Throughput by Satisfaction:\n", throughput_by_sat)

plt.figure(figsize=(6,4))
df.boxplot(column="latency_ms", by="satisfaction_label")
plt.title("Latency by Customer Satisfaction Level")
plt.suptitle("")
plt.xlabel("Satisfaction Level")
plt.ylabel("Latency (Standardized)")
plt.show()

# =========================================================
# STEP 7: Descriptive Statistics by District
# =========================================================
district_stats = df.groupby("district")[num_cols].agg(
    ["mean", "std", "min", "max"]
)

print("\nDescriptive Statistics by District:")
print(district_stats)

# =========================================================
# STEP 8: Machine Learning Classification
# =========================================================
X = df[num_cols]
y = df["satisfaction_label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# STEP 9: Define and Train Models
# =========================================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", probability=True, random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "model": model
    }

# =========================================================
# STEP 10: Model Comparison Results
# =========================================================
print("\n================ MODEL PERFORMANCE =================\n")

best_model_name = ""
best_acc = 0

for name, res in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print("Classification Report:\n", res["report"])
    print("Confusion Matrix:\n", res["cm"])
    print("-"*60)

    if res["accuracy"] > best_acc:
        best_acc = res["accuracy"]
        best_model_name = name

print(f"\n✅ BEST MODEL: {best_model_name} (Accuracy = {best_acc:.4f})")

# =========================================================
# STEP 11: Confusion Matrix Visualization
# =========================================================
for name, res in results.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(
        res["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dissatisfied","Neutral","Satisfied"],
        yticklabels=["Dissatisfied","Neutral","Satisfied"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =========================================================
# STEP 12: Predict on Full Dataset
# =========================================================
best_model = results[best_model_name]["model"]
df["satisfaction_predicted"] = best_model.predict(X)

# Save final dataset
df.to_excel(
    "DATA/Telecom_Service_Quality_Rwanda_2023_2025_with_predictions.xlsx",
    index=False
)

print("\n✅ Pipeline completed successfully!")
