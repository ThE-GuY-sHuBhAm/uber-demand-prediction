# 🚖 NYC Uber Demand Prediction (End-to-End MLOps Pipeline)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)

## Project Overview

This project is a complete **End-to-End MLOps Pipeline** for predicting Uber/Taxi demand in New York City. It forecasts the number of pickups in specific regions of the city for the next 15-minute interval.

Unlike standard "notebook" projects, this is built as a production-grade system with **Data Versioning (DVC)**, **Experiment Tracking (MLflow)**, and **Scalable Data Processing (Dask)**.

## Dataset Information
The project utilizes the NYC Yellow Taxi Trip Data.
- Source: http://kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data

---

## The Challenge: Supply vs. Demand

The core objective of this project is to minimize the mismatch between the number of drivers available (Supply) and passengers looking for a ride (Demand).

### 1. Understanding the Raw Patterns

We analyze historical data to track driver availability and user requests hour-by-hour. As seen below, both follow distinct temporal patterns.

![Supply vs Demand Patterns](references/images/supply_demand_1.PNG)
*Figure 1: Hourly trends for Drivers vs. Users around Central Park.*

### 2. Identifying the Mismatch

When we overlay these trends, the business problem becomes clear:
- **Excess of Drivers:** Leads to wasted time and lower driver earnings.
- **Lack of Drivers:** Leads to surge pricing, longer wait times, and lost customers.

![Supply Demand Mismatch](references/images/supply_demand_2.PNG)
*Figure 2: Visualizing the gap where demand exceeds supply (opportunity for prediction) vs. supply exceeds demand.*

**Our Goal:** Accurately predict these demand spikes so drivers can be repositioned *before* the shortage happens.

---

## Key Results

- **Final Model:** LightGBM Regressor (Log-Transformed Target)
- **Performance:** **MAPE: 25.8%** (Beat baseline by ~5%)
- **Architecture:** 5-Stage DVC Pipeline (Ingest → Split → Feature Eng → Train → Evaluate)

---

### 1. Data Ingestion & Splitting

- **Tech:** `Dask`, `Pandas`
- **Logic:** Ingests raw NYC Taxi data (Millions of rows)
- **Leakage Prevention:** We perform a strict **Time-Based Split** (Jan-Feb for Training, March for Testing) *before* any processing to prevent "Look-Ahead Bias"

### 2. Spatial Clustering

- **Tech:** `MiniBatchKMeans`, `Scikit-Learn`
- **Logic:** Converts raw Latitude/Longitude into **30 Discrete Regions**
- **Optimization:** We found that **30 clusters** was the "sweet spot" (MAPE 0.258) compared to 50 clusters (MAPE 0.303)

### 3. Feature Engineering

- **Tech:** `Pandas`, `NumPy`
- **Features:**
  - **Lags:** Demand from t-1 to t-4 (Previous 1 hour)
  - **Rolling Window:** 1-hour moving average to smooth noise
  - **Time:** Hour of Day, Day of Week, Weekend Flag
- **Transformation:** Applied `np.log1p` to the target variable to handle the "Zero-Inflated" nature of demand data (many quiet regions with 0-1 pickups)

### 4. Model Training

- **Tech:** `LightGBM`, `XGBoost`
- **Configuration:** Gradient Boosting Decision Tree (GBDT) with Log-Link function
- **Tracking:** All experiments (params, metrics, artifacts) are logged to **MLflow (DagsHub)**

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ThE-GuY-sHuBhAm/uber-demand-prediction.git
cd uber-demand-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup DVC & MLflow

Configure your remote storage (if using S3/DagsHub):

```bash
# Set your MLflow tracking URI (export or .env)
export MLFLOW_TRACKING_URI="https://dagshub.com/ThE-GuY-sHuBhAm/uber-demand-prediction.mlflow"
```

### 4. Run the Pipeline

Reproduce the entire experiment from raw data to final model:

```bash
dvc repro
```

---

## Project Structure

```
├── .dvc/                  # DVC configuration
├── data/                  # Data storage (Git-ignored)
│   ├── raw/               # Raw NYC Taxi CSVs
│   ├── interim/           # Split data (Train/Test Raw)
│   └── processed/         # Final features for training
├── models/                # Saved models (Scaler, K-Means, Encoder, LightGBM)
├── src/                   # Source code
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── split_data.py
│   ├── features/
│   │   ├── extract_features.py   # Clustering & Resampling
│   │   └── feature_processing.py # Lags & Rolling features
│   └── models/
│       ├── train.py              # LightGBM Training
│       ├── evaluate.py           # MLflow Logging
│       └── register_model.py     # Model Registry
├── dvc.yaml               # DVC Pipeline Definition
├── params.yaml            # Hyperparameters (Clusters, Smoothing, Model Params)
└── requirements.txt       # Python dependencies
```

---

## Experiments & Learnings

| Experiment | Model | Key Change | MAPE Score | Status |
|------------|-------|------------|------------|--------|
| **Exp 1** | Linear Regression | Baseline | 0.323 | Underfitting |
| **Exp 2** | XGBoost | Non-Linearity Added | 0.308 | Better |
| **Exp 3** | LightGBM (Optuna) | Hyperparameter Tuning | 0.298 |  Overfitting |
| **Exp 4** | **LightGBM (Log-Transform)** | **Log(Target + 1)** | **0.258** |  **Best** |

### Critical Fix: Data Leakage

Early versions of the model had an artificially low MAPE (~8%) because K-Means clustering was trained on the *entire* dataset (including the Test set). We fixed this by strictly fitting the Scaler and K-Means models **only on the Training set (Jan-Feb)** and transforming the Test set (March) using the saved artifacts. This resulted in a realistic and robust error metric of ~25%.

---

## Model Performance

The final LightGBM model with log transformation achieved:
- **MAPE:** 25.8%
- **Improvement over baseline:** ~5 percentage points
- **Key insight:** Log transformation helped handle the zero-inflated nature of demand data

---

## Technologies Used

- **Data Processing:** Dask, Pandas, NumPy
- **Machine Learning:** LightGBM, XGBoost, Scikit-Learn
- **MLOps:** DVC (Data Version Control), MLflow (Experiment Tracking)
- **Optimization:** Optuna (Hyperparameter Tuning)
- **Clustering:** MiniBatchKMeans

---

## License

This project is open source and available under the [MIT License](LICENSE).


---

## Acknowledgments
- DagsHub for MLflow hosting
- The open-source community for amazing tools like DVC, MLflow, and LightGBM
