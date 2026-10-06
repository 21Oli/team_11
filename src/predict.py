"""
Reusable prediction pipeline for the crop-yield project.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cleaning import basic_cleaning, load_data
from features import (
    get_feature_columns,
    prepare_test_features,
)


TARGET_COLUMN = "yield_tons_per_ha"


def load_model(model_path):
    """
    Load a trained sklearn pipeline.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict(
    model,
    test_df,
    feature_columns=None,
):
    """
    Generate predictions.

    Parameters
    ----------
    model : fitted sklearn pipeline
    test_df : pandas.DataFrame
    feature_columns : list, optional
        Exact feature list used during training.

    Returns
    -------
    numpy.ndarray
    """
    test_df = basic_cleaning(test_df)

    if feature_columns is None:
        feature_columns = get_feature_columns(test_df)

    X_test = prepare_test_features(
        test_df,
        feature_columns,
    )

    predictions = model.predict(X_test)

    predictions = np.asarray(predictions)

    if len(predictions) != len(test_df):
        raise ValueError(
            "Number of predictions does not match test rows."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    return predictions


def create_submission(
    model_path,
    test_path,
    submission_template_path,
    output_path,
):
    """
    Generate the final hackathon submission.
    """
    # Load model
    model = load_model(model_path)

    # Load test data
    test_df = load_data(test_path)
    test_df = basic_cleaning(test_df)

    # Get the feature list from the fitted pipeline.
    #
    # The training pipeline contains the original
    # ColumnTransformer, so the transformer columns provide
    # the exact features used during training.
    preprocessor = model.named_steps["preprocessor"]

    feature_columns = []

    for _, _, columns in preprocessor.transformers_:
        if columns == "drop":
            continue

        if columns == "remainder":
            continue

        feature_columns.extend(list(columns))

    # Remove duplicates while preserving order
    feature_columns = list(
        dict.fromkeys(feature_columns)
    )

    # Generate predictions
    predictions = predict(
        model=model,
        test_df=test_df,
        feature_columns=feature_columns,
    )

    # Load official submission template
    template = pd.read_csv(
        submission_template_path
    )

    if "plot_id" not in template.columns:
        raise ValueError(
            "Submission template must contain 'plot_id'."
        )

    if len(template) != len(predictions):
        raise ValueError(
            "Submission template row count does not "
            "match prediction count."
        )

    # Create submission
    submission = template[["plot_id"]].copy()

    submission[
        "predicted_yield_tons_per_ha"
    ] = predictions

    # Required checks
    if not submission["plot_id"].is_unique:
        raise ValueError(
            "Submission contains duplicate plot IDs."
        )

    if submission[
        "predicted_yield_tons_per_ha"
    ].isna().any():
        raise ValueError(
            "Submission contains missing predictions."
        )

    if not np.isfinite(
        submission[
            "predicted_yield_tons_per_ha"
        ]
    ).all():
        raise ValueError(
            "Submission contains invalid predictions."
        )

    # Save
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print("Submission created successfully.")
    print("Rows:", len(submission))
    print("Unique plot IDs:", submission["plot_id"].nunique())
    print("Saved to:", output_path.resolve())

    return submission


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "models"
        / "final_model.joblib"
    )

    test_path = (
        project_root
        / "data"
        / "processed"
        / "master_test.csv"
    )

    submission_template_path = (
        project_root
        / "data"
        / "raw"
        / "submission_template.csv"
    )

    output_path = (
        project_root
        / "submission"
        / "team_submission.csv"
    )

    create_submission(
        model_path=model_path,
        test_path=test_path,
        submission_template_path=submission_template_path,
        output_path=output_path,
    )