#STEP 2: Data Cleaning
# ===============================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import geopandas as gpd
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')
data_path = (r"C:\Users\auguc\Documents\GitHub\project DSCI  COHORT1\DATA\telecom_qos_dataset_2023_70rows.csv"
)

df = pd.read_csv(data_path)
# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month


# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# ===============================
# STEP 1.3: Encoding categorical variables
# ===============================

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['satisfaction_label_encoded'] = le.fit_transform(df['satisfaction_label'])
mapping = {'Poor': 0, 'Neutral': 1, 'Good': 2}
df['satisfaction_label_encoded'] = df['satisfaction_label'].map(mapping)



print("\nDataset after encoding:")
print(df.head())
#Check numeric columns for scaling (optional for ML models)
from sklearn.preprocessing import StandardScaler

num_cols = ['downlink_throughput_kbps','uplink_throughput_kbps','latency_ms',
            'jitter_ms','packet_loss_pct','rsrp_dbm','rsrq_db','cell_load_pct',
            'active_users','call_setup_success_rate_cssr_pct','drop_call_rate_dcr_pct',
            'mos_voice','customer_reported_qoe_score','Data_Usage_MB']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.to_csv('DATA/telecom_qos_dataset_cleaned.csv', index=False)


#Distribution of Key QoS Indicators
import matplotlib.pyplot as plt

plt.hist(df["latency_ms"], bins=30)
plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Network Latency")
plt.show()
df.columns
# ===============================
# STEP 4.1: Correlation Analysis
# ===============================

# Select numerical QoS features
corr_features = [
    'downlink_throughput_kbps',
    'uplink_throughput_kbps',
    'latency_ms',
    'jitter_ms',
    'packet_loss_pct',
    'rsrp_dbm',
    'rsrq_db',
    'cell_load_pct',
    'active_users',
    'call_setup_success_rate_cssr_pct',
    'drop_call_rate_dcr_pct',
    'mos_voice',
    'customer_reported_qoe_score',
    'Data_Usage_MB'
]

# Compute correlation matrix
corr_matrix = df[corr_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of QoS Indicators")
plt.show()
# Show only strong correlations
strong_corr = corr_matrix[(corr_matrix.abs() >= 0.7)]
print(strong_corr)
# ===============================
# STEP 4.2: Satisfaction vs QoS Comparison
# ===============================

# Compare average latency by satisfaction level
latency_by_satisfaction = df.groupby('satisfaction_label')['latency_ms'].mean()
print("Average Latency by Satisfaction Level:")
print(latency_by_satisfaction)

# Compare average throughput by satisfaction level
throughput_by_satisfaction = df.groupby('satisfaction_label')['downlink_throughput_kbps'].mean()
print("\nAverage Downlink Throughput by Satisfaction Level:")
print(throughput_by_satisfaction)
### visualisation 
plt.figure(figsize=(6,4))
df.boxplot(column='latency_ms', by='satisfaction_label')
plt.title("Latency Distribution by Customer Satisfaction Level")
plt.suptitle("")
plt.xlabel("Satisfaction Level")
plt.ylabel("Latency (standardized)")
plt.show()


# ===============================
# STEP 4.3: Descriptive Statistics by District
# ===============================

# Select key QoS indicators
qos_features = [
    'downlink_throughput_kbps',
    'uplink_throughput_kbps',
    'latency_ms',
    'jitter_ms',
    'packet_loss_pct',
    'rsrp_dbm',
    'rsrq_db',
    'cell_load_pct',
    'mos_voice',
    'customer_reported_qoe_score'
]

# Group by district and compute descriptive statistics (mean, std, min, max)
district_stats = df.groupby('district')[qos_features].agg(
    ['mean', 'std', 'min', 'max']
)

# Display results
print("Descriptive Statistics of QoS Indicators by District:")
print(district_stats)
