# 🌾 AI-Powered Agricultural Yield Prediction

An end-to-end **AI & Data Engineering** project for predicting crop yield in Ethiopia using historical agricultural, farm-management, weather, and market data.

The project combines data cleaning and integration, exploratory analysis, visualization, machine learning, model evaluation, and an interactive prediction demo.

---

## 🚀 Live Demo

**Demo:** `YOUR_STREAMLIT_DEMO_URL`

> Replace `YOUR_STREAMLIT_DEMO_URL` with the deployed Streamlit application URL.

---

## 🎯 Problem

Agricultural productivity is affected by multiple factors including:

* Weather conditions
* Rainfall
* Temperature
* Soil quality
* Fertilizer usage
* Improved seed adoption
* Pest and disease conditions
* Farm characteristics
* Access to markets

The goal of this project is to use historical data to estimate **crop yield in tons per hectare (`yield_tons_per_ha`)** and provide an accessible decision-support interface.

---

## 💡 Solution

We developed an end-to-end machine-learning pipeline that:

1. Cleans and standardizes heterogeneous agricultural data.
2. Integrates farm, weather, and price information.
3. Performs data-quality and integrity checks.
4. Conducts exploratory analysis.
5. Produces decision-oriented visualizations.
6. Compares multiple machine-learning models.
7. Performs cross-validation and temporal validation.
8. Performs weather-feature ablation.
9. Performs hyperparameter tuning.
10. Saves the final trained model.
11. Provides an interactive Streamlit prediction application.

---

## 🏗️ Project Structure

```text
.
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── assets/
│       ├── cleaned_weather.csv
│       ├── cleaned_price.csv
│       └── final_model.joblib
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── master_train.csv
│       └── master_test.csv
│
├── figures/
│   ├── *.png
│   └── ...
│
├── models/
│   └── final_model.joblib
│
├── notebooks/
│   ├── 01_cleaning_and_integration.ipynb
│   ├── 02_analysis_report.ipynb
│   ├── 03_visualizations.ipynb
│   └── 04_modeling_and_evaluation.ipynb
│
├── reports/
│   ├── A_*.md
│   ├── B_analysis_report.*
│   ├── D_model_evaluation.md
│   ├── model_results.csv
│   ├── validation_predictions.csv
│   └── feature_importance.csv
│
├── submission/
│   └── team_submission.csv
│
└── README.md
```

---

## 📊 Dataset

The integrated modeling dataset contains agricultural plot-level observations with variables covering:

### Farm and crop information

* Region
* Crop type
* Survey year
* Planting month
* Farm size
* Altitude
* Fertilizer application
* Improved seed usage
* Pest/disease status
* Soil quality
* Labor requirements
* Distance to market

### Weather information

* Seasonal rainfall
* Seasonal mean temperature
* Total seasonal rainfall
* Extreme heat days
* Temperature variability
* Temperature deviation
* Weather-month availability
* Missing weather months

### Engineered features

* Planting month number
* Fertilizer × improved-seed interaction
* Rainfall per fertilizer unit

### Target

```text
yield_tons_per_ha
```

The `plot_id` is an identifier and is not used as a predictive feature.

---

## 🔬 Data Pipeline

The project follows a structured A–E workflow.

### A — Cleaning & Integration

The pipeline performs:

* Data cleaning
* Standardization
* Source joining
* Join auditing
* Join validation
* Feature-table construction
* Integrity checks
* Train/test export
* Data dictionary generation

Main outputs:

```text
data/processed/master_train.csv
data/processed/master_test.csv
```

---

### B — Analysis

The analysis notebook contains the required analytical tasks covering:

* Dataset structure
* Target distribution
* Group-level comparisons
* Missingness
* Relationships between variables
* Agricultural and weather patterns
* Findings relevant to modeling and decision-making

Output:

```text
reports/B_analysis_report.*
```

---

### C — Visualizations

The visualization notebook generates the required **12 PNG figures** with captions.

Output:

```text
figures/
```

---

### D — Modeling & Evaluation

The modeling pipeline includes:

* **D1:** Baselines
* **D2:** Model comparison
* **D3:** 5-fold cross-validation
* **D4:** Out-of-time validation
* **D5:** Weather ablation
* **D6:** Hyperparameter tuning
* **D7:** Error analysis
* **D8:** Response to findings
* **D9:** Plain-language metric explanation

The final model is saved as:

```text
models/final_model.joblib
```

Additional evaluation outputs:

```text
reports/D_model_evaluation.md
reports/model_results.csv
reports/validation_predictions.csv
reports/feature_importance.csv
```

---

## 🤖 Machine Learning

Multiple model families were evaluated, including:

* Mean baseline
* Linear Regression
* Random Forest
* Extra Trees
* HistGradientBoosting

Model selection considered:

* RMSE
* MAE
* R²
* Cross-validation stability
* Temporal generalization
* Training time
* Reproducibility

The final selected model is used directly by the demo application.

---

## 📏 Evaluation Metrics

### RMSE

Root Mean Squared Error measures the typical magnitude of prediction errors while giving larger errors greater weight.

```text
RMSE = √mean((y - ŷ)²)
```

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted yield.

```text
MAE = mean(|y - ŷ|)
```

### R²

R² measures how much of the variation in the target is explained by the model.

A value closer to `1` indicates stronger explanatory performance, while values around or below `0` indicate that the model is not outperforming a simple mean-based reference on that evaluation split.

---

## 🌦️ Weather Contribution

Weather features were explicitly evaluated through an ablation experiment.

The model was compared:

```text
With weather features
        vs.
Without weather features
```

This helps determine whether weather information contributes meaningful predictive value rather than assuming that it does.

---

## 🧪 Validation Strategy

The project uses multiple validation perspectives:

### Random validation

Used for standard model comparison.

### 5-fold cross-validation

Used to evaluate model stability across different training/validation partitions.

### Out-of-time validation

The model is evaluated on a later year after training on earlier years.

This provides a more realistic test of temporal generalization.

---

## 🌾 Interactive Demo

The Streamlit application allows users to enter agricultural and seasonal conditions and receive an estimated crop yield.

The demo uses:

```text
app/assets/final_model.joblib
```

The application does **not retrain the model**.

It provides:

* Crop selection
* Region selection
* Farm characteristics
* Fertilizer information
* Improved seed information
* Pest/disease information
* Soil quality
* Weather conditions
* Predicted yield
* Estimated total farm production
* Reference market-price information

### Run locally

Install the application dependencies:

```bash
pip install -r app/requirements.txt
```

Start the application:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## 📦 Demo Assets

The application bundles the required reference data and trained model:

```text
app/assets/
├── cleaned_weather.csv
├── cleaned_price.csv
└── final_model.joblib
```

The market-price reference information is used for contextual information in the application and is **not passed to the prediction model**.

---

## 📤 Submission

The final prediction submission is available at:

```text
submission/team_submission.csv
```

The submission contains the predicted yield for the required test plots.

---

## 👥 Team Workflow

The project was developed collaboratively using GitHub with separate responsibilities across:

* Data cleaning & integration
* Analysis & visualization
* Modeling & evaluation
* Application & presentation

All work was integrated into a single reproducible project structure.

---

## ⚠️ Limitations

This model provides an estimate based on historical observations and should not be interpreted as a guaranteed agricultural outcome.

Prediction quality may be affected by:

* Limited historical observations
* Missing weather information
* Regional differences
* Unobserved farm-management practices
* Extreme weather events outside the historical range
* Changes in agricultural practices over time

The model should therefore be used as a **decision-support tool**, not as a replacement for agronomic expertise or field-level assessment.

---

## 🔮 Future Improvements

Potential future extensions include:

* Satellite-derived vegetation indices
* Soil and remote-sensing data
* More granular weather forecasts
* Crop-specific models
* Uncertainty intervals
* Explainable-AI dashboards
* Seasonal forecasting
* Regional early-warning systems
* Integration with agricultural advisory platforms

---

## 🛠️ Technology Stack

```text
Python
Pandas
NumPy
Scikit-learn
Joblib
Matplotlib
Streamlit
Jupyter
Git
GitHub
```

---

## 🏆 Hackathon Deliverables

| Deliverable                | Status                  |
| -------------------------- | ----------------------- |
| A — Cleaning & Integration | ✅ Complete              |
| B — Analysis               | ✅ Complete              |
| C — Visualizations         | ✅ Complete              |
| D — Modeling & Evaluation  | ✅ Complete              |
| E — Interactive Demo       | 🚀 Ready for deployment |

---

## 📄 License

This project was developed as part of an AI & Data Engineering hackathon.
