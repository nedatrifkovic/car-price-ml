import json
from pathlib import Path
from typing import Dict, List, Union
import joblib
import pandas as pd
import numpy as np

# Paths to the trained model and feature configuration
MODEL_PATH = Path("models/car_price_model_XGBoost_Tuned.pkl")
FEATURE_CONFIG_PATH = Path("models/feature_config.json")


class ModelBundle:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        config_path: Path = FEATURE_CONFIG_PATH,
    ) -> None:
        # Load the trained model (Pipeline with preprocessor + XGBRegressor)
        self.model = joblib.load(model_path)
        # Load feature configuration
        self.config = json.loads(config_path.read_text())
        self.feature_order: List[str] = self.config["all_features"]

    def _to_dataframe(
        self, record_or_records: Union[Dict, List[Dict]]
    ) -> pd.DataFrame:
        """
        Convert a dict or list of dicts into a DataFrame
        with all required feature columns in the correct order.
        """
        if isinstance(record_or_records, dict):
            data = [record_or_records]
        else:
            data = record_or_records

        df = pd.DataFrame(data)

        # Add missing columns with NaN and keep column order
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_order]

        return df

    def predict(
        self, record_or_records: Union[Dict, List[Dict]]
    ) -> List[float]:
        """
        Run predictions on a dict or list of dicts.
        Returns a list of predicted prices.
        """
        df = self._to_dataframe(record_or_records)
        preds = self.model.predict(df)
        return preds.tolist()

    def predict_df(
        self, record_or_records: Union[Dict, List[Dict]]
    ) -> pd.DataFrame:
        """
        Run predictions and return a DataFrame
        with the original inputs + predicted_price column.
        """
        df = self._to_dataframe(record_or_records)
        df["predicted_price"] = self.model.predict(df)
        return df


if __name__ == "__main__":
    # Example local test
    bundle = ModelBundle()
    example = {
        "manufacturer": "Toyota",
        "model": "Corolla",
        "prod_year": 2015,
        "category": "Sedan",
        "leather_interior": "No",
        "fuel_type": "Petrol",
        "engine_volume": 1.6,
        "mileage": 120000,
        "cylinders": 4.0,
        "gear_box_type": "Automatic",
        "drive_wheels": "front",
        "doors": 4,
        "wheel": "Left wheel",
        "color": "White",
        "airbags": 6,
        "is_Turbo": "No",
    }

    df_preds = bundle.predict_df(example)
    print(df_preds)
