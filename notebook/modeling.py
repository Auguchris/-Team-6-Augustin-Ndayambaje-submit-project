# =========================================================
# MODEL TRAINING AND EVALUATION
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Load Cleaned Dataset
# ---------------------------------------------------------
df = pd.read_excel(
    "DATA/Telecom_Service_Quality_Rwanda_2023_2025_cleaned.xlsx"
)

# ---------------------------------------------------------
# 2. Define Features and Target Variable
# ---------------------------------------------------------
features = [
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

X = df[features]
y = df["satisfaction_label_encoded"]

# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 4. Feature Scaling
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. Initialize Models
# ---------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}

# ---------------------------------------------------------
# 6. Train and Evaluate Models
# ---------------------------------------------------------
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

# ---------------------------------------------------------
# 7. Display Results
# ---------------------------------------------------------
print("\n================ MODEL PERFORMANCE =================\n")

best_model = ""
best_accuracy = 0

for name, res in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print("Classification Report:\n", res["classification_report"])
    print("Confusion Matrix:\n", res["confusion_matrix"])
    print("-" * 60)

    if res["accuracy"] > best_accuracy:
        best_accuracy = res["accuracy"]
        best_model = name

print(f"\n✅ Best Performing Model: {best_model} "
      f"(Accuracy = {best_accuracy:.4f})")
# ---------------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# ---------------------------------------------------------
for name, res in results.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(
        res["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dissatisfied", "Neutral", "Satisfied"],
        yticklabels=["Dissatisfied", "Neutral", "Satisfied"]
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()
