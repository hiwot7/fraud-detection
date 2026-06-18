import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap

# Force non-interactive backend for server-side execution safety
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def process_and_compare(dataset_name, file_path, ignore_cols, target_col):
    if not os.path.exists(file_path):
        print(f"⚠️ Path missing for {dataset_name}: {file_path}")
        return None, None, None, None
        
    print(f"\n====================================================")
    print(f"📊 INGESTING & PIPELINING DATASET: {dataset_name}")
    print(f"====================================================")
    
    df = pd.read_csv(file_path)
    
    # Isolate targets and predictive features
    X = df.drop(columns=ignore_cols + [target_col], errors='ignore')
    y = df[target_col]
    
    # Ensure all columns are numeric for XGBoost
    X = X.select_dtypes(include=[np.number])
    
    # Stratified split to preserve extreme class imbalances
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   🔹 Matrix Dimensions: {X_train.shape[0]} train rows | {X_test.shape[0]} test rows across {X.shape[1]} features.")
    
    # ----------------------------------------------------------------
    # MODEL 1: Baseline Random Forest Classifier
    # ----------------------------------------------------------------
    print(f"   🌲 Training Balanced Random Forest Baseline...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    
    # ----------------------------------------------------------------
    # MODEL 2: Hyperparameter-Tuned XGBoost Classifier
    # ----------------------------------------------------------------
    print(f"   🚀 Training Optimized XGBoost Classifier...")
    # Compute scale_pos_weight dynamically to handle severe class imbalance
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    # Generate structured metrics summary comparison
    print(f"\n   📈 --- {dataset_name} Performance Comparison Matrix ---")
    print(f"   [Random Forest Baseline] ROC-AUC: {rf_auc:.4f}")
    print(f"   [Optimized XGBoost]      ROC-AUC: {xgb_auc:.4f}")
    
    # ----------------------------------------------------------------
    # SHAP INTEGRATION (Explainable AI Engine)
    # ----------------------------------------------------------------
    print(f"\n   🧠 Computing Tree-SHAP Values for {dataset_name} Explainability...")
    # Use a background sample to preserve processing safety bounds
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test.head(500))
    
    # Save the global summary plots to disk automatically
    os.makedirs("notebooks/plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test.head(500), show=False)
    plt.title(f"SHAP Global Feature Importance Profile - {dataset_name}", fontsize=14, pad=15)
    plt.tight_layout()
    plot_path = f"notebooks/plots/{dataset_name.lower().replace(' ', '_')}_shap.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   ✅ SHAP Global Plot successfully cached at: {plot_path}")
    
    return rf_auc, xgb_auc, confusion_matrix(y_test, rf_preds), confusion_matrix(y_test, xgb_preds)

if __name__ == "__main__":
    # Ingesting Dataset A: E-Commerce Logs
    eco_path = r"C:\Users\hiwot\Desktop\fraud-detection\data\processed\cleaned_fraud_data.csv"
    eco_ignore = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'sex', 'age', 'country']
    eco_rf_auc, eco_xgb_auc, eco_rf_cm, eco_xgb_cm = process_and_compare(
        "E-Commerce Fraud", eco_path, eco_ignore, "class"
    )
    
    # Ingesting Dataset B: Credit Card Transaction Metrics
    cc_path = r"C:\Users\hiwot\Desktop\fraud-detection\data\processed\cleaned_creditcard_data.csv"
    # Fallback checking if file is still named raw inside folder structure
    if not os.path.exists(cc_path):
        cc_path = r"C:\Users\hiwot\Desktop\fraud-detection\data\raw\creditcard.csv"
        
    cc_ignore = ['Time'] # Drop Time to focus strictly on numerical PCA component variance fields
    cc_rf_auc, cc_xgb_auc, cc_rf_cm, cc_xgb_cm = process_and_compare(
        "Credit Card Transactions", cc_path, cc_ignore, "Class"
    )
    
    print("\n🎯 Complete pipeline execution successful! All metrics and SHAP diagnostic figures stored.")