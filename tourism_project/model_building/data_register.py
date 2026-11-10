import pandas as pd
import os

# Define the path to the dataset
DATA_PATH = "tourism_project/data/tourism.csv"

# Define the expected columns
EXPECTED_COLUMNS = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

def register_data():
    """
    Reads the dataset, performs basic validation, and prints a summary.
    """
    print("--- Starting Data Registration ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully from {DATA_PATH}")

    # Validate that all expected columns are present
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    print("All expected columns are present.")

    # Print a short summary
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData types and non-null counts:")
    df.info()

    print("--- Data Registration Complete ---")


if __name__ == "__main__":
    register_data()
