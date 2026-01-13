# =========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Cleaned Dataset
# -------------------------------
df = pd.read_excel(
    "DATA/Telecom_Service_Quality_Rwanda_2023_2025_cleaned.xlsx"
)

# -------------------------------
# 2. Distribution of Key QoS Indicators
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(df["latency_ms"], bins=30)
plt.xlabel("Latency (Standardized)")
plt.ylabel("Frequency")
plt.title("Distribution of Network Latency")
plt.show()

# -------------------------------
# 3. Correlation Analysis
# -------------------------------
qos_features = [
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

corr_matrix = df[qos_features].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of QoS Indicators")
plt.show()

# -------------------------------
# 4. QoS vs Customer Satisfaction
# -------------------------------
latency_by_satisfaction = df.groupby("satisfaction_label")["latency_ms"].mean()
throughput_by_satisfaction = df.groupby("satisfaction_label")[
    "downlink_throughput_kbps"
].mean()

print("\nAverage Latency by Satisfaction Level:\n", latency_by_satisfaction)
print("\nAverage Downlink Throughput by Satisfaction Level:\n", throughput_by_satisfaction)

# Boxplot
plt.figure(figsize=(6,4))
df.boxplot(column="latency_ms", by="satisfaction_label")
plt.title("Latency Distribution by Satisfaction Level")
plt.suptitle("")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Latency (Standardized)")
plt.show()

# -------------------------------
# 5. Descriptive Statistics by District
# -------------------------------
district_stats = df.groupby("district")[qos_features].agg(
    ["mean", "std", "min", "max"]
)

print("\nDescriptive Statistics by District:")
print(district_stats)

print("\n✅ EDA completed successfully.")
