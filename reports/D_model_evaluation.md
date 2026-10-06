# D — Modeling & Evaluation

## D1 — Baselines

Validation split: 80% training / 20% validation.

Random seed: 42.

### Mean Baseline

- RMSE: 1.4131
- MAE: 1.1042
- R²: -0.0001

### Linear Regression

- RMSE: 0.8887
- MAE: 0.6653
- R²: 0.6044

## D2 — Model Comparison

The compared nonlinear model families were Random Forest, Extra Trees,
and HistGradientBoosting.

| model                |     rmse |      mae |           r2 |   training_time_sec |
|:---------------------|---------:|---------:|-------------:|--------------------:|
| HistGradientBoosting | 0.477949 | 0.345593 |  0.885581    |           2.78136   |
| Random Forest        | 0.527944 | 0.38617  |  0.860392    |           1.88268   |
| Extra Trees          | 0.537597 | 0.392942 |  0.855241    |           1.1191    |
| Linear Regression    | 0.888704 | 0.665279 |  0.604407    |           0.0960174 |
| Mean baseline        | 1.41305  | 1.10423  | -0.000113367 |           0.0268234 |

Selected initial winner: **HistGradientBoosting**.

The winner was selected using validation RMSE while also considering
MAE, R², training time, and reproducibility.

## D3 — Cross-Validation

Five-fold cross-validation was performed for the selected model and
the runner-up.

| model                |   cv_rmse_mean |   cv_rmse_std |
|:---------------------|---------------:|--------------:|
| HistGradientBoosting |       0.474241 |    0.0117826  |
| Random Forest        |       0.526219 |    0.00794658 |

The standard deviation of fold RMSE indicates the stability of model
performance across different training/validation partitions.

## D4 — Out-of-Time Check

The model was trained using 2021–2023 and evaluated on 2024.

- RMSE: 0.4960
- MAE: 0.3622
- R²: 0.8804

This provides a temporal generalization check against the random split.

## D5 — Weather Ablation

### With Weather

- RMSE: 0.4779
- MAE: 0.3456
- R²: 0.8856

### Without Weather

- RMSE: 0.5277
- MAE: 0.3797
- R²: 0.8605

Weather RMSE improvement: 0.0498

Relative improvement: 9.43%

## D6 — Hyperparameter Tuning

A GridSearchCV search was performed using a documented search grid.

Number of configurations: 24

Best parameters:

```text
{'model__max_depth': None, 'model__max_features': 0.8, 'model__min_samples_leaf': 1, 'model__n_estimators': 250}
```
