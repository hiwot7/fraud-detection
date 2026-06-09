import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load the raw datasets from your validated data/raw/ folder
print("🔄 Loading datasets...")
try:
    fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")
    credit_df = pd.read_csv("data/raw/creditcard.csv")
    print("✅ All raw datasets successfully loaded into memory.")
except FileNotFoundError as e:
    print(f"❌ Error locating data files. Please check paths. Details: {e}")
    exit()

# 2. Convert Data Types (Timestamps to datetime objects)
print("⏳ Processing temporal attributes...")
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

# 3. Vectorized IP Address to Integer Conversion for Range Lookup


def ip_to_int(ip_str):
    try:
        if isinstance(ip_str, (int, float)):
            return int(ip_str)
        # Handle standard string dot-notation IP parsing
        parts = list(map(int, ip_str.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return 0


# Convert IP column format to integer metrics dynamically
if fraud_df['ip_address'].dtype == 'object':
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
else:
    fraud_df['ip_int'] = fraud_df['ip_address'].astype(float).astype(int)

# 4. Geolocation Integration using specialized # 4. Geolocation Integration using specialized Interval Matching Lookup
print("🌍 Performing IP range-to-country geolocation mapping...")

# Ensure both lookup keys are exactly the same data type (int64) to prevent MergeError
fraud_df['ip_int'] = fraud_df['ip_int'].astype('int64')
ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(
    'int64')
ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(
    'int64')

# Sort both dataframes to fulfill merge_asof requirements
fraud_df = fraud_df.sort_values('ip_int')
ip_df = ip_df.sort_values('lower_bound_ip_address')

# Perform interval backward matching lookup
merged_fraud_df = pd.merge_asof(
    fraud_df,
    ip_df,
    left_on='ip_int',
    right_on='lower_bound_ip_address',
    direction='backward'
)

# 5. Feature Engineering
print("🛠️ Engineering behavioral and temporal features...")
# Time elapsed since initial user signup in fractional hours
merged_fraud_df['time_since_signup'] = (
    merged_fraud_df['purchase_time'] - merged_fraud_df['signup_time']
).dt.total_seconds() / 3600.0

# Extract explicit cyclical temporal traits
merged_fraud_df['hour_of_day'] = merged_fraud_df['purchase_time'].dt.hour
merged_fraud_df['day_of_week'] = merged_fraud_df['purchase_time'].dt.dayofweek

# 6. Save the High-Performance Processed Outputs
print("💾 Saving clean structures to disk...")
merged_fraud_df.to_csv("data/processed/cleaned_fraud_data.csv", index=False)
credit_df.to_csv("data/processed/cleaned_creditcard_data.csv", index=False)

print("\n🚀 ✅ Preprocessing and Geolocation Mapping Complete!")
print("Outputs are now securely stored inside your 'data/processed/' directory.")
