"""
Feature-selection and feature-preparation utilities
for the crop-yield prediction project.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "yield_tons_per_ha"
ID_COLUMNS = {"plot_id"}

WEATHER_KEYWORDS = [
    "weather",
    "temp",
    "temperature",
    "rain",
    "rainfall",
    "extreme_heat",
    "heat_days",
    "season",
]


def find_price_columns(df):
    """
    Identify price-related columns.

    Price must not be used as a model feature.
    """
    return [
        column
        for column in df.columns
        if "price" in column.lower()
    ]


def find_weather_features(df):
    """
    Identify weather-derived candidate features.
    """
    features = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in WEATHER_KEYWORDS
        )
    ]

    return [
        column
        for column in features
        if column != TARGET_COLUMN
    ]


def get_feature_columns(df):
    """
    Return model features while excluding:

    - target
    - plot ID
    - price-related columns
    """
    price_columns = find_price_columns(df)

    forbidden = {
        TARGET_COLUMN,
        *ID_COLUMNS,
        *price_columns,
    }

    features = [
        column
        for column in df.columns
        if column not in forbidden
    ]

    return features


def prepare_features(df):
    """
    Separate features and target.

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series
    feature_columns : list
    weather_features : list
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    feature_columns = get_feature_columns(df)
    weather_features = find_weather_features(df)

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    # Challenge requirement: at least one weather-derived feature
    # must be available to the model.
    weather_in_features = [
        column
        for column in weather_features
        if column in feature_columns
    ]

    if not weather_in_features:
        raise ValueError(
            "No weather-derived feature is present in the model features."
        )

    # Safety check: price must never enter the model.
    price_in_features = [
        column
        for column in feature_columns
        if "price" in column.lower()
    ]

    if price_in_features:
        raise ValueError(
            f"Price leakage detected: {price_in_features}"
        )

    return X, y, feature_columns, weather_in_features


def prepare_test_features(df, feature_columns):
    """
    Prepare test features using exactly the training feature list.

    This prevents train/test feature mismatch.
    """
    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Test data is missing features: {missing}"
        )

    return df[feature_columns].copy()


def build_preprocessor(X):
    """
    Build the reusable sklearn preprocessing pipeline.
    """
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor