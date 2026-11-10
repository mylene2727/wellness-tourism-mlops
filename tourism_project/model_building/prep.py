"""
Data Preparation Script
-----------------------
This script performs data cleaning, feature engineering, and train/test split
for the Wellness Tourism Package prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ----- Configuration -----
DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "."  # Splits will be saved in the current directory (repo root)

def load_data():
    """Load raw dataset."""
    print("[1/5] Loading raw data...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def clean_data(df):
    """Clean the dataset: fix typos, drop irrelevant columns, handle NaNs."""
    print("[2/5] Cleaning data...")

    # Fix the 'Fe Male' typo in the Gender column (data quality issue)
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    # Drop CustomerID: it has no predictive power (unique identifier)
    df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

    # Impute missing values for categorical columns with the mode (most frequent)
    cat_cols_with_na = ['Gender', 'MaritalStatus', 'TypeofContact']
    for col in cat_cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Impute missing values for numerical columns with the median
    # (median is more robust to outliers than the mean)
    num_cols_with_na = [
        'Age', 'DurationOfPitch', 'NumberOfFollowups',
        'PreferredPropertyStar', 'NumberOfTrips',
        'NumberOfChildrenVisiting', 'MonthlyIncome'
    ]
    for col in num_cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"      Remaining missing values: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    """Create new features to help the model."""
    print("[3/5] Engineering features...")

    # Feature 1: Total number of people on the trip (adults + children)
    df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']

    # Feature 2: Average spend per trip (income proxy per trip)
    # Add 1 to avoid division by zero
    df['IncomePerTrip'] = df['MonthlyIncome'] / (df['NumberOfTrips'] + 1)

    # Feature 3: Pitch efficiency (satisfaction per minute of pitch)
    # Add 1 to avoid division by zero
    df['PitchEfficiency'] = df['PitchSatisfactionScore'] / (df['DurationOfPitch'] + 1)

    print(f"      Created 3 new features. New shape: {df.shape}")
    return df


def split_and_save(df):
    """Split data into train/test and save to CSV."""
    print("[4/5] Splitting data into train/test (80/20, stratified)...")

    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    # stratify=y ensures the same proportion of buyers/sellers in train and test.
    # random_state=42 ensures reproducibility (same split every run).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"      Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"      Train target distribution:\n{y_train.value_counts(normalize=True)}")

    # Save splits to CSV (they will be uploaded as GitHub Actions artifacts)
    X_train.to_csv(f"{OUTPUT_DIR}/Xtrain.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/Xtest.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/ytrain.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/ytest.csv", index=False)
    print(f"[5/5] Splits saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    split_and_save(df)
    print("--- Data Preparation Complete ---")
