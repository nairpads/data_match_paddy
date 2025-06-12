import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

st.title("Train Reconciliation Match Prediction Model")

# Step 1: Upload Training Data
st.header("Upload Labeled Training Data")
train_file = st.file_uploader("Upload a CSV with labeled debit-credit pairs", type="csv")

if train_file is not None:
    df = pd.read_csv(train_file)
    st.success("Training file uploaded successfully!")
    st.write(df.head())

    # Clean and validate target column
    df = df.dropna(subset=['match_label'])
    df['match_label'] = df['match_label'].astype(int)

    # Feature engineering
    df['amount_diff'] = abs(df['debit_amount'] - df['credit_amount'])
    df['date_diff'] = abs((pd.to_datetime(df['debit_date'], dayfirst=True, errors='coerce') - pd.to_datetime(df['credit_date'], dayfirst=True, errors='coerce')).dt.days)
    df['narration_score'] = df.apply(lambda row: fuzz.partial_ratio(str(row['debit_narration']), str(row['credit_narration'])), axis=1)
    df['dr_cr_pair'] = df.apply(lambda row: 1 if row.get('debit_dc_flag') == 'D' and row.get('credit_dc_flag') == 'C' else 0, axis=1)

    # NEW: Combined score feature
    df['combined_score'] = df['narration_score'] - df['amount_diff'] - df['date_diff']

    X = df[['amount_diff', 'date_diff', 'narration_score', 'dr_cr_pair', 'combined_score']]
    y = df['match_label']

    # Show class distribution
    st.subheader("Class Balance")
    st.write(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation")
    st.text(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "reconciliation_model.pkl")
    st.success("Model trained and saved as 'reconciliation_model.pkl'")
