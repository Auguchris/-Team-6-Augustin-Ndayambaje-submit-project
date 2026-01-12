
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------
# Step 1: Load your cleaned dataset
# -------------------------------
# Replace path with your cleaned CSV if needed
pf = pd.read_csv("DATA/telecom_qos_dataset_cleaned.csv")

# -------------------------------
# Step 2: Features and Target
# -------------------------------
features = [
    'downlink_throughput_kbps', 'uplink_throughput_kbps', 'latency_ms',
    'jitter_ms', 'packet_loss_pct', 'rsrp_dbm', 'rsrq_db',
    'cell_load_pct', 'active_users', 'call_setup_success_rate_cssr_pct',
    'drop_call_rate_dcr_pct', 'mos_voice', 'customer_reported_qoe_score',
    'Data_Usage_MB', 'hour', 'day', 'month'
]

X = pf[features]  # ✅ Corrected (use DataFrame, not pd)
y = pf['satisfaction_label_encoded']

# -------------------------------
# Step 3: Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 4: Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# FULL ML SCRIPT CORRECTED: Random Forest + Logistic Regression + SVM
# ===============================

# --------------- Step 1: Import Libraries -----------------
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# --------------- Step 2: Load Dataset -----------------
data_path = r"C:\Users\auguc\Documents\GitHub\project DSCI  COHORT1\DATA\telecom_qos_dataset_cleaned.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at {data_path}")

pf = pd.read_csv(data_path)
print("Dataset loaded successfully!\n")
print(pf.head())

# --------------- Step 3: Define Features & Target -----------------
features = [
    'downlink_throughput_kbps', 'uplink_throughput_kbps', 'latency_ms',
    'jitter_ms', 'packet_loss_pct', 'rsrp_dbm', 'rsrq_db',
    'cell_load_pct', 'active_users', 'call_setup_success_rate_cssr_pct',
    'drop_call_rate_dcr_pct', 'mos_voice', 'customer_reported_qoe_score',
    'Data_Usage_MB', 'hour', 'day', 'month'
]

X = pf[features]
y = pf['satisfaction_label_encoded']

# --------------- Step 4: Train/Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------- Step 5: Feature Scaling -----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------- Step 6: Define Models -----------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

# --------------- Step 7: Train, Predict & Evaluate -----------------
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_test)
    # Store the trained model too
    results[name] = {
        "model": model,
        "accuracy": acc,
        "classification_report": classification_report(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "predictions_test": y_pred_test
    }

# --------------- Step 8: Compare Model Performance -----------------
print("\n================ MODEL PERFORMANCE COMPARISON ================\n")
best_acc = 0
best_model_name = ""

for name, res in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print("Classification Report:\n", res['classification_report'])
    print("Confusion Matrix:\n", res['confusion_matrix'])
    print("-"*60)
    
    # Check which model is best
    if res['accuracy'] > best_acc:
        best_acc = res['accuracy']
        best_model_name = name

print(f"\n✅ Best Model: {best_model_name} with Accuracy = {best_acc:.4f} → Good Performance")

# --------------- Step 9: Plot Confusion Matrix for all models -----------------
for name, res in results.items():
    plt.figure(figsize=(6,4))
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Poor','Neutral','Good'],
                yticklabels=['Poor','Neutral','Good'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.show()


# --------------- Step 10: Predict on Full Dataset -----------------
best_model = results[best_model_name]['model']
X_scaled_all = scaler.transform(pf[features])
pf['satisfaction_predicted'] = best_model.predict(X_scaled_all)

# --------------- Step 11: Export Predictions -----------------
export_folder = r"C:\Users\auguc\Documents\GitHub\project DSCI  COHORT1\DATA"
os.makedirs(export_folder, exist_ok=True)
export_path = os.path.join(export_folder, "telecom_qos_dataset_predictions_best_model.csv")

pf.to_csv(export_path, index=False)
print(f"\nPredictions exported successfully to {export_path}")