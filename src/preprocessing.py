import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Default path for preprocessed data 
# (can be overridden by environment variable)
DEFAULT_PROCESSED_PATH = os.getenv(
    "PROCESSED_DATA_PATH", "data/processed/car_data_processed.csv"
)

# Columns that look like indexes and should be removed if present
INDEX_LIKE_COLS = {"Unnamed: 0", "index", "ID", "id"}


def load_preprocessed_data(path: str = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    """
    Loads already preprocessed data from a CSV file.
    - Default path: data/processed/car_data_processed.csv
    - Automatically removes index-like columns if they exist
    """
    df = pd.read_csv(path)

    drop_cols = [c for c in df.columns if c in INDEX_LIKE_COLS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def infer_feature_types(
    df: pd.DataFrame, target: str
) -> tuple[list[str], list[str]]:
    """
    Automatically determine categorical and numerical features.
    - Target column is excluded
    - Object/Category -> categorical
    - Other dtypes -> numerical
    """
    df_ = df.drop(columns=[target]) if target in df.columns else df.copy()
    cat = df_.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in df_.columns if c not in cat]
    
    return cat, num


def build_preprocessor(
    categorical_features: list[str], numeric_features: list[str]
) -> ColumnTransformer:
    """
    Creates a ColumnTransformer:
    - OneHotEncoder for categorical features
    - StandardScaler for numeric features
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Splits the dataset into train and test sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
