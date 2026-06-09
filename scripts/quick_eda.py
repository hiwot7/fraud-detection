import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 Generating emergency EDA insights...")
df = pd.read_csv("data/processed/cleaned_fraud_data.csv")

# 1. Class Distribution
print("\n--- Fraud Class Distribution ---")
print(df['class'].value_counts(normalize=True) * 100)

# 2. Top High-Risk Countries
print("\n--- Top 5 Countries with Most Fraud Volume ---")
fraud_cases = df[df['class'] == 1]
print(fraud_cases['country'].value_counts().head(5))

# 3. Top High-Risk Browsers
print("\n--- Fraud Count by Browser ---")
print(fraud_cases['browser'].value_counts())

print("\n✨ Done! Copy these text statistics directly into your submission text box or README!")
