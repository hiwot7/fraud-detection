import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_and_evaluate():
    # 1. Define paths
    INPUT_PATH = r"C:\Users\hiwot\Desktop\fraud-detection\data\processed\engineered_fraud_data.csv"
    
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Cannot find engineered matrix at: {INPUT_PATH}")
        
    print(f"🌲 Loading engineered dataset: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Identify and drop non-predictive tracking columns
    # We drop metadata/strings because algorithms can only process mathematical vectors
    ignore_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'sex', 'age', 'country']
    drop_targets = [col for col in ignore_cols if col in df.columns]
    
    # Separate features (X) and ground truth target label (y)
    X = df.drop(columns=drop_targets + ['class'], errors='ignore')
    y = df['class']
    
    print(f"📊 Extracted Feature Matrix shape: {X.shape[1]} input signals for modeling.")
    
    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    # Stratify ensures both splits retain the exact same fraud-to-genuine balance ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"💾 Data Split Complete: {X_train.shape[0]} training examples | {X_test.shape[0]} evaluation targets.")
    
    # 4. Instantiate and Train the Classifier
    print("\n🏋️ Training Random Forest Classifier with Cost-Sensitive Class Balancing...")
    # class_weight='balanced' automatically penalizes errors on the rare fraud class heavily
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   ✅ Model training sequence finished.")
    
    # 5. Generate Predictions and Evaluation Metrics
    print("\n🔍 Running test matrix inference evaluations...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 6. Print Production-Grade Diagnostics Report
    print("\n======= 📈 MACHINE LEARNING METRICS SUMMARY =======")
    print(classification_report(y_test, y_pred, target_names=['Genuine (0)', 'Fraudulent (1)']))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score (Area Under Curve Matrix): {roc_auc:.4f}")
    
    print("\n======= 🧩 CONFUSION MATRIX TRUTH SIGNATURES =======")
    cm = confusion_matrix(y_test, y_pred)
    print(f" True Negatives  (Correctly flagged Genuine): {cm[0][0]}")
    print(f" False Positives (Genuine flagged as Fraud) : {cm[0][1]}")
    print(f" False Negatives (Missed Fraud Attacks)     : {cm[1][0]}")
    print(f" True Positives  (Successfully Caught Fraud): {cm[1][1]}")
    print("====================================================")

if __name__ == "__main__":
    train_and_evaluate()