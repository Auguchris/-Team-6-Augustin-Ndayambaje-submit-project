# =========================================================
# DATA CLEANING & PREPROCESSING
# =========================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load Dataset
# -------------------------------
data_path = r"C:\Users\auguc\Documents\GitHub\project DSCI  COHORT1\DATA\Telecom_Service_Quality_Rwanda_2023_2025.xlsx"

df = pd.read_excel(data_path)

# -------------------------------
# 2. Standardize Column Names
# -------------------------------
df.columns = df.columns.str.strip().str.lower()

# -------------------------------
# 3. Basic Dataset Inspection
# -------------------------------
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# -------------------------------
# 4. Remove Duplicates
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# 5. Encode Satisfaction Labels
# -------------------------------
label_mapping = {
    "Dissatisfied": 0,
    "Neutral": 1,
    "Satisfied": 2
}

df["satisfaction_label_encoded"] = df["satisfaction_label"].map(label_mapping)

print("\nLabel Encoding Check:")
print(df[["satisfaction_label", "satisfaction_label_encoded"]].head())

# -------------------------------
# 6. Feature Scaling (Numerical)
# -------------------------------
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

# -------------------------------
# 7. Save Cleaned Dataset
# -------------------------------
df.to_excel(
    "DATA/Telecom_Service_Quality_Rwanda_2023_2025_cleaned.xlsx",
    index=False
)

print("\n✅ Data cleaning completed successfully.")
