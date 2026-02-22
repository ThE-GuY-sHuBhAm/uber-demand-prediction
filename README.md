# 🚖 Spatiotemporal Taxi Demand Forecasting & Driver Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## Project Overview

This project is a **production-style spatiotemporal forecasting system** that predicts taxi demand across New York City at a regional level and recommends optimal areas for drivers to reposition in real time.

Unlike notebook-only ML projects, this system is built as a full **MLOps pipeline** with:

- DVC for reproducible pipelines  
- MLflow for experiment tracking & model registry  
- Scalable preprocessing using Dask  
- Causal time-series forecasting (leakage-free)  
- Real-time driver recommendation app (Streamlit)

---

## Business Problem

Urban ride-hailing platforms constantly suffer from **supply-demand imbalance**:

- Too many drivers → idle time & lost income  
- Too few drivers → surge pricing & poor user experience  

### Understanding the Raw Patterns

We analyze historical data to track driver availability and user requests hour-by-hour. As seen below, both follow distinct temporal patterns.

![Supply vs Demand Patterns](references/images/supply_demand_1.PNG)
*Figure 1: Hourly trends for Drivers vs. Users around Central Park.*

### Identifying the Mismatch

When we overlay these trends, the business problem becomes clear:
- **Excess of Drivers:** Leads to wasted time and lower driver earnings.
- **Lack of Drivers:** Leads to surge pricing, longer wait times, and lost customers.

![Supply Demand Mismatch](references/images/supply_demand_2.PNG)
*Figure 2: Visualizing the gap where demand exceeds supply (opportunity for prediction) vs. supply exceeds demand.*

**Our Goal:** Predict **regional demand for the next 15-minute interval** and guide drivers toward high-probability pickup zones.

---

## 📊 Dataset

NYC Yellow Taxi Trip Records  
Source: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data  

Millions of pickup events across NYC.

---

## ⚙️ System Architecture (Leakage-Free)

```
Raw Data
↓
Outlier Cleaning (Dask)
↓
Spatial Clustering (MiniBatch K-Means)
↓
15-Minute Demand Resampling
↓
Lag & Rolling Features
↓
Next-Step Forecast Target (t+1)
↓
Temporal Train/Test Split
↓
LightGBM Forecasting Model
↓
MLflow Registry
↓
Streamlit Driver Recommendation App
```

---

## 🧠 Key Modeling Decisions

### 📍 Spatial Regions

Converted GPS coordinates into **30 optimized demand zones** using MiniBatch K-Means.

30 clusters provided the best trade-off between:
- Spatial precision  
- Driver reachability  
- Predictive stability  

---

### ⏱ Temporal Forecasting

Predicts:

> **Demand in the next 15 minutes**

Using:

- Lag features (last 1 hour)  
- Rolling demand trend  
- Time-of-day & weekday patterns  

---

### 📈 Model

**LightGBM Regressor (log-transformed target)**  
Chosen for:

✔ Non-linear demand behavior  
✔ High performance on tabular time-series features  

---

## ✅ Final Performance (Leakage-Free)

| Metric | Value |
|--------|-------|
| **MAPE** | **31.9%** |
| Baseline | ~40% |
| Improvement | ~8% |

> Earlier versions achieved lower error due to temporal leakage — fully corrected in the final pipeline.

---

## 🚕 Driver Recommendation System (Streamlit)

The deployed app:

- Predicts next-interval demand for all regions  
- Ranks zones by expected pickups  
- Filters to nearby reachable regions  
- Visually highlights best relocation choices  

This converts forecasting into **real operational decisions**.

---

## 🔁 Reproducible Pipeline (DVC)

Run the entire system from raw data to deployed model:

```bash
dvc repro
```

All parameters, artifacts, and experiments are version-tracked.

---

## 🧪 Experiment History

| Experiment | Model | Change | MAPE |
|------------|-------|--------|------|
| Exp 1 | Linear Regression | Baseline | 0.40 |
| Exp 2 | XGBoost | Non-linear | 0.35 |
| Exp 3 | LightGBM | Boosted Trees | 0.33 |
| Exp 4 | LightGBM + Log Target | Final Model | **0.319** |

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Data Engineering** | Dask, Pandas, NumPy |
| **Machine Learning** | LightGBM, XGBoost, Scikit-Learn |
| **MLOps** | DVC, MLflow (DagsHub) |
| **Visualization** | Streamlit |
| **Clustering** | MiniBatch K-Means |

---

## 📁 Project Structure

```
data/
 ├─ raw/
 ├─ interim/
 └─ processed/

models/
src/
dvc.yaml
params.yaml
app.py
```

---

## 🚀 Key Learnings

✔ Preventing temporal leakage is critical in forecasting  
✔ Causal feature design improves real-world reliability  
✔ Spatial clustering enables scalable urban modeling  
✔ MLOps pipelines turn ML into reproducible systems  
