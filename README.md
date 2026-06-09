# 🛡️ Fraud Detection System for E-commerce and Banking

An end-to-end fraud detection pipeline designed to identify fraudulent activities in e-commerce transactions and bank credit card utilization. This system integrates advanced geolocation mapping, temporal feature engineering, and robust machine learning workflows to handle highly imbalanced datasets.

---

## 📂 Project Structure

```text
fraud-detection/
├── .gitignore                  # Prevents tracking of heavy data and environment folders
├── README.md                   # Main project documentation
├── requirements.txt            # Python environment dependencies
├── data/                       # [HIDDEN BY GITIGNORE - Data directory]
│   ├── raw/                    # Contains raw: Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv
│   └── processed/              # Stores the engineered and geolocation-mapped outputs
├── notebooks/
│   ├── eda-fraud-data.ipynb    # Exploratory Data Analysis for e-commerce transactions
│   └── eda-creditcard.ipynb    # Exploratory Data Analysis for credit card utilization
└── scripts/
    ├── .gitkeep
    └── preprocess.py           # Core automated preprocessing pipeline script
