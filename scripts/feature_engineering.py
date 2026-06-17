import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads the cleaned dataset from disk safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data file not found at: {file_path}")
    print(f"🔄 Loading data for feature transformation from: {file_path}")
    return pd.read_csv(file_path)

def engineering_pipeline():
    # 1. Setup exact paths
    INPUT_PATH = r"C:\Users\hiwot\Desktop\fraud-detection\data\processed\cleaned_fraud_data.csv"
    OUTPUT_DIR = r"C:\Users\hiwot\Desktop\fraud-detection\data\processed"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "engineered_fraud_data.csv")
    
    df = load_data(INPUT_PATH)
    
    print("🛠️ Initiating mathematical feature extraction calculations...")
    
    # 2. Temporal Features (Time Deltas between actions)
    # JS-to-Python Check: Converting raw timestamp objects and computing differences in seconds
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['time_to_purchase_seconds'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        print("   ✅ Temporal calculation complete: 'time_to_purchase_seconds' generated.")
    
    # 3. Behavioral Frequency Tracking (Transaction Velocity)
    # Counts how many times a unique user ID appears across the shopping system
    if 'user_id' in df.columns:
        df['user_transaction_velocity'] = df.groupby('user_id')['user_id'].transform('count')
        print("   ✅ Velocity profiling complete: 'user_transaction_velocity' generated.")
        
    # 4. Categorical Transformation (One-Hot Encoding for the Gateway Models)
    # JS-to-Python Check: This is identical to converting categorical string flags into binary 1/0 properties
    categorical_targets = ['browser', 'source']
    available_categories = [col for col in categorical_targets if col in df.columns]
    
    if available_categories:
        df = pd.get_dummies(df, columns=available_categories, drop_first=True, dtype=int)
        print(f"   ✅ One-hot encoding complete for columns: {available_categories}")

    # 5. Numerical Feature Scaling (Standardization mapping to a mean of 0)
    # Protects distance-sensitive components from being overwhelmed by large values
    scaler = StandardScaler()
    scale_targets = ['purchase_value']
    if 'time_to_purchase_seconds' in df.columns:
        scale_targets.append('time_to_purchase_seconds')
        
    print(f"   ⚖️ Standardizing scales for metrics: {scale_targets}")
    df[scale_targets] = scaler.fit_transform(df[scale_targets])
    
    # 6. Save the structured matrix back to storage
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎯 Feature Engineering Pipeline complete! Transformed dataset saved at:\n   -> {OUTPUT_FILE}")
    print(f"📊 Final Data Grid Matrix: {df.shape[0]} rows across {df.shape[1]} engineered features.")

if __name__ == "__main__":
    engineering_pipeline()