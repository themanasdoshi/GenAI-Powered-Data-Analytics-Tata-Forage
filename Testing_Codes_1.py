import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_delinquency_model():
    """
    Loads delinquency data from CSV, performs cleaning and imputation, 
    engineers features, trains a Random Forest Classifier, and evaluates performance.
    """
    print("Starting Delinquency Prediction Model Pipeline...")

    # --- 1. DATA LOADING ---
    # Assuming the converted CSV file is named 'Delinquency_prediction_dataset.csv'
    file_path = "Delinquency_prediction_dataset.csv" 
    
    try:
        # Using read_csv for the converted file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded file: {file_path}")
    except FileNotFoundError:
        print(f"ERROR: Data file '{file_path}' not found. Please ensure the CSV is in the same directory.")
        return
    
    # --- 2. DATA CLEANING & IMPUTATION ---
    
    # Standardize Employment Status for consistency
    df['Employment_Status'] = df['Employment_Status'].replace({
        'EMP': 'Employed', 
        'employed': 'Employed', 
        'Self-employed': 'Self_Employed'
    })

    # Impute Income (Synthetic Normal Distribution - preserves variance)
    np.random.seed(42)
    income_mean = df['Income'].mean()
    income_std = df['Income'].std()
    null_income_mask = df['Income'].isnull()
    
    # Generate and fill random values for missing income
    df.loc[null_income_mask, 'Income'] = np.random.normal(income_mean, income_std, size=null_income_mask.sum())
    df['Income'] = df['Income'].clip(lower=0) # Ensure no negative income

    # Impute Loan_Balance and Credit_Score with Median (robust to outliers)
    df['Loan_Balance'] = df['Loan_Balance'].fillna(df['Loan_Balance'].median())
    df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].median())

    # --- 3. FEATURE ENGINEERING & ENCODING ---
    
    # Map ordinal values for Month history ('On-time'=0, 'Late'=1, 'Missed'=2)
    month_mapping = {'On-time': 0, 'Late': 1, 'Missed': 2}
    month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']

    for col in month_cols:
        df[col + '_Num'] = df[col].map(month_mapping)

    # One-Hot Encoding for nominal categorical variables
    df_encoded = pd.get_dummies(df, columns=['Employment_Status', 'Location', 'Credit_Card_Type'], drop_first=True)

    # Define the complete list of features (X)
    features = [
        'Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments', 
        'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure'
    ] + [col + '_Num' for col in month_cols] + [
        c for c in df_encoded.columns if 'Employment_' in c or 'Location_' in c or 'Credit_Card_' in c
    ]
    
    X = df_encoded[features]
    y = df_encoded['Delinquent_Account']
    
    print(f"Total features selected for modeling: {len(features)}")

    # --- 4. MODEL TRAINING & SPLIT ---
    
    # Stratified split to maintain delinquency ratio in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Random Forest Classifier with 'balanced' class weight to handle imbalance
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10 
    )
    print("Training Random Forest Classifier...")
    rf_model.fit(X_train, y_train)

    # --- 5. EVALUATION ---
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1] 

    print("\n" + "="*40)
    print("      DELINQUENCY MODEL PERFORMANCE")
    print("="*40)
    
    # Report provides Precision, Recall, and F1-score
    print("\nClassification Report (Optimized for Recall):")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature Importance (Explainability)
    importances = pd.DataFrame({
        'Feature': features, 
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Top 5 Feature Importances (Risk Indicators) ---")
    print(importances.head(5).to_markdown(index=False))
    print("="*40)
    print("Pipeline Complete.")

if __name__ == "__main__":
    run_delinquency_model()