import json
import os
from pathlib import Path

import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.preprocessing import (
    load_preprocessed_data,
    infer_feature_types,
    build_preprocessor,
    split_data,
)  # End of pipeline steps

# Paths
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
TARGET_COL = "price"


def evaluate_and_print(name, model, X_test, y_test):
    """Helper function to evaluate models and print metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} -> MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R²: {r2:,.4f}")
    return mae, rmse, r2


def train_model():
    # Load preprocessed dataset
    df = load_preprocessed_data()

    # Identify categorical and numeric features
    categorical_features, numeric_features = infer_feature_types(
        df, TARGET_COL
    )

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    # Define base models
    base_models = {
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    }

    # Define hyperparameter search space for each model
    param_distributions = {
        "DecisionTree": {
            "regressor__max_depth": [5, 10, 20, None],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__max_features": ["sqrt", "log2", None],
        },
        "RandomForest": {
            "regressor__n_estimators": [200, 500, 800],
            "regressor__max_depth": [10, 20, 30, None],
            "regressor__max_features": ["sqrt", "log2"],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "regressor__n_estimators": [500, 1000, 1500],
            "regressor__learning_rate": [0.01, 0.05, 0.1],
            "regressor__max_depth": [4, 6, 8],
            "regressor__subsample": [0.7, 0.8, 1.0],
            "regressor__colsample_bytree": [0.7, 0.8, 0.9],
            "regressor__reg_alpha": [0, 0.1, 0.5],
            "regressor__reg_lambda": [1, 1.5, 2],
        },
    }

    results = {}
    best_model = None
    best_r2 = float("-inf")

    # Train and tune each model
    for name, model in base_models.items():
        print(f"\n🔹 Training model: {name}")

        # Build pipeline: preprocessing + regression model
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )

        # Run randomized hyperparameter search with cross-validation
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions[name],
            n_iter=15,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        search.fit(X_train, y_train)

        tuned_model = search.best_estimator_

        # Evaluate tuned model
        mae, rmse, r2 = evaluate_and_print(
            f"{name}_Tuned", tuned_model, X_test, y_test
        )

        results[name] = {
            "model": tuned_model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        # Save best model (based on R²)
        if r2 > best_r2:
            best_r2 = r2
            best_model = (name, tuned_model)

    # Save best model to disk
    if best_model:
        model_name, model_obj = best_model
        model_path = MODELS_DIR / (f"car_price_model_{model_name}_Tuned.pkl")
        joblib.dump(model_obj, model_path)
        print(f"\n✅ Best model ({model_name}_Tuned) saved -> {model_path}")

    # Save feature configuration
    feature_config = {
        "target": TARGET_COL,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "all_features": categorical_features + numeric_features,
        "data_path": os.getenv(
            "PROCESSED_DATA_PATH",
            "data/processed/car_data_processed.csv",
        ),
    }
    FEATURE_CONFIG_PATH.write_text(json.dumps(feature_config, indent=2))
    print(f"✅ Feature config saved -> {FEATURE_CONFIG_PATH}")


if __name__ == "__main__":
    train_model()
