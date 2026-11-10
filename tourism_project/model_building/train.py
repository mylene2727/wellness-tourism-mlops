"""
Model Training Script
---------------------
This script trains, tunes, evaluates and saves multiple ML models
for the Wellness Tourism Package purchase prediction.
MLflow is used for experiment tracking.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
)
import xgboost as xgb
import mlflow
import mlflow.sklearn

# ----- Configuration -----
TRAIN_X = "Xtrain.csv"
TRAIN_Y = "ytrain.csv"
TEST_X = "Xtest.csv"
TEST_Y = "ytest.csv"
MODEL_OUTPUT = "tourism_project/deployment/best_model.joblib"
RANDOM_STATE = 42


def load_splits():
    """Load train and test splits."""
    print("[1/6] Loading train/test splits...")
    X_train = pd.read_csv(TRAIN_X)
    X_test = pd.read_csv(TEST_X)
    y_train = pd.read_csv(TRAIN_Y).squeeze("columns")
    y_test = pd.read_csv(TEST_Y).squeeze("columns")
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """
    Build a ColumnTransformer that:
    - imputes + scales numeric columns
    - imputes + one-hot encodes categorical columns
    This guarantees NO data leakage: the pipeline is fit on training data only.
    """
    print("[2/6] Building preprocessing pipeline...")

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"      Numeric columns ({len(num_cols)}): {num_cols}")
    print(f"      Categorical columns ({len(cat_cols)}): {cat_cols}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    return preprocessor


def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train 3 models, tune hyperparameters via GridSearchCV,
    log to MLflow, and return the best one.
    """
    print("[3/6] Defining models and hyperparameter grids...")

    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__solver': ['liblinear', 'lbfgs']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            ),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__subsample': [0.8, 1.0]
            }
        }
    }

    # MLflow: point to local tracking folder
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("wellness_tourism_purchase_prediction")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("[4/6] Training and tuning models...")
    for name, config in models.items():
        print(f"\n  → Training {name} ...")

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])

        with mlflow.start_run(run_name=name):

            grid = GridSearchCV(
                pipeline,
                param_grid=config['params'],
                cv=cv,
                scoring='f1',       # tune on F1 (imbalanced target)
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_

            # Predictions on test set
            y_pred = best.predict(X_test)
            y_proba = best.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }

            # Log to MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best, artifact_path="model")

            results[name] = {'best_model': best, 'metrics': metrics, 'best_params': grid.best_params_}
            print(f"      Best params: {grid.best_params_}")
            print(f"      Metrics: {metrics}")

    return results


def pick_best(results):
    """Select the best model by F1-score."""
    print("[5/6] Selecting best model by F1-score...")
    best_name = max(results, key=lambda k: results[k]['metrics']['f1'])
    print(f"      Best model: {best_name} (F1 = {results[best_name]['metrics']['f1']:.4f})")
    return best_name, results[best_name]


def save_model(name, result):
    """Save the best pipeline to disk."""
    print("[6/6] Saving best model...")
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    joblib.dump(result['best_model'], MODEL_OUTPUT)
    print(f"      Model saved to {MODEL_OUTPUT}")
    # Also save metadata for traceability
    meta_path = MODEL_OUTPUT.replace('.joblib', '_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"Model: {name}\n")
        f.write(f"Best params: {result['best_params']}\n")
        f.write(f"Metrics: {result['metrics']}\n")
    print(f"      Metadata saved to {meta_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_splits()
    preprocessor = build_preprocessor(X_train)
    results = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor)
    best_name, best_result = pick_best(results)

    # Final evaluation report for the best model
    y_pred = best_result['best_model'].predict(X_test)
    print(f"\n===== Final Evaluation — {best_name} =====")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    save_model(best_name, best_result)
    print("\n--- Model Training Complete ---")
