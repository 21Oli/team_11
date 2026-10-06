"""
Reusable training pipeline for the crop-yield project.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from cleaning import basic_cleaning, load_data
from features import (
    build_preprocessor,
    prepare_features,
)


RANDOM_STATE = 42

TARGET_COLUMN = "yield_tons_per_ha"


def build_model(X):
    """
    Build the complete preprocessing + model pipeline.
    """
    preprocessor = build_preprocessor(X)

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(
    train_path,
    model_path,
):
    """
    Train and save the final model pipeline.

    Parameters
    ----------
    train_path : str or Path
        Training CSV path.

    model_path : str or Path
        Output .joblib path.

    Returns
    -------
    pipeline
        Fitted sklearn pipeline.
    """
    # Load
    df = load_data(train_path)

    # Clean
    df = basic_cleaning(df)

    # Prepare
    X, y, feature_columns, weather_features = prepare_features(df)

    print("Training rows:", len(X))
    print("Number of features:", len(feature_columns))
    print("Weather features:", weather_features)

    # Build pipeline
    pipeline = build_model(X)

    # Train
    pipeline.fit(X, y)

    # Save
    model_path = Path(model_path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(pipeline, model_path)

    print()
    print("Model trained successfully.")
    print("Saved to:", model_path.resolve())

    return pipeline


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    train_path = (
        project_root
        / "data"
        / "processed"
        / "master_train.csv"
    )

    model_path = (
        project_root
        / "models"
        / "final_model.joblib"
    )

    train_model(
        train_path=train_path,
        model_path=model_path,
    )