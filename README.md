<div align="center">

# ☀️ Smart Renewable Energy Guardian

**Predict renewable energy output, monitor performance, and catch abnormal deviations, with an explanation for every prediction.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-EB5E28)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-8A2BE2)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![R2](https://img.shields.io/badge/Test%20R%C2%B2-0.937-2ea44f)

</div>

---

## 📌 Overview

Most energy-forecasting projects stop at "train a model, report a score."
**Smart Renewable Energy Guardian** goes further. It treats the model as the core of a small monitoring system:

| # | Capability | What it does |
|---|-----------|--------------|
| 1 | **Energy Prediction** | Predicts expected energy output (Wh) from weather and time conditions |
| 2 | **Performance Monitoring** | Compares actual vs. expected production and tracks deviation |
| 3 | **Anomaly Detection** | Flags hours where output is far from what the weather justifies |
| 4 | **What-if Simulation** | Lets the user change GHI, temperature or cloud cover and see the effect |
| 5 | **Explainable AI (SHAP)** | Shows *why* the model predicted a given value |
| 6 | **Interactive Dashboard** | One Streamlit app that ties everything together |

> **Scenario:** The weather says the plant should produce ~800 Wh, but it produced only 350 Wh.
> The system computes the deviation, marks the hour as *Underperformance*, and shows which weather factors shaped the expectation, so an operator knows it needs review.

---

## 🖥️ Dashboard Pages

| Page | Purpose |
|------|---------|
| **Home** | System summary: monitored hours, anomaly count and rate, status distribution |
| **Energy Prediction** | Predicted output from weather sliders, plus a check against actual production |
| **Anomaly Detection** | Actual vs. predicted timeline per month, anomalies highlighted, worst cases table |
| **What-if Simulation** | Before/after prediction and curves showing the effect of GHI and cloud cover |
| **Model Insights** | Model metrics, feature importance, SHAP explanation for the current scenario |

---

## 📊 Results

**Dataset:** 196,776 records, 17 original columns (energy output + weather + sun features), time-ordered.

### Model comparison (test set, 80/20 split)

| Model | Train R² | Test R² | MAE (Wh) | RMSE (Wh) |
|-------|:-------:|:------:|:-------:|:--------:|
| Decision Tree | 0.954 | 0.934 | 102 | 270 |
| Random Forest | 0.982 | 0.949 | 90.0 | 237 |
| XGBoost | 0.967 | 0.950 | 89.5 | 233 |
| LightGBM | 0.963 | 0.950 | 90.5 | 234 |
| **XGBoost (weather-only, deployed)** | 0.952 | **0.937** | 110 | 262 |

### Why the deployed model drops the lag features

The best scores above use `energy_lag_1..3` (energy from previous steps). That makes the model lean on *past production*, so it cannot be used to judge whether current production is normal, and it flattens What-if simulations.

The **deployed model uses only weather and time features**. It loses about 0.013 R², and in exchange:

- predictions depend on conditions only, so a deviation from them is meaningful;
- What-if simulation responds properly to weather changes;
- the system stays honest about what drives production.

### Anomaly detection

- Evaluated on **daytime hours with expected output > 100 Wh** (17,115 hours).
- An hour is flagged when `|actual − predicted| / predicted > 70%`.
- Result: **1,415 flagged hours (8.27%)**, split into *Underperformance* and *Overperformance*.
- The threshold was chosen by sweeping values (50 → 100) and keeping the flag rate in the 5–10% range.

| Threshold | Flagged hours | Rate |
|:--------:|:-----------:|:----:|
| 50% | 2,893 | 16.90% |
| 60% | 2,017 | 11.78% |
| **70%** | **1,415** | **8.27%** |
| 80% | 1,026 | 5.99% |
| 100% | 393 | 2.30% |

### What drives the predictions (SHAP)

Top features by mean |SHAP|: `GHI` (by a wide margin), `GHI_normalized`, `dayLength`, `SunlightTime/daylength`, `day_of_year_cos`, `clouds_all`.
Solar irradiance dominates, and the effect of cloud cover is largely captured through GHI, which matches the physics of solar generation.

---

## 🧱 Project Pipeline

```
Raw data ─► Cleaning & EDA ─► Feature engineering ─► Model comparison
                                                        │
                      ┌─────────────────────────────────┘
                      ▼
          Weather-only XGBoost ─► Anomaly detection ─► What-if ─► SHAP ─► Streamlit
```

**Feature engineering:** cyclical encoding of hour / month / day-of-year, GHI interactions (with temperature and humidity), cloud-effect index, clear-sky indicator, precipitation total, wind–temperature interaction.

---

## 🚀 Run Locally

```bash
git clone https://github.com/<your-username>/smart-renewable-energy-guardian.git
cd smart-renewable-energy-guardian

pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure

```
├── app.py                    # Streamlit dashboard (5 pages)
├── requirements.txt
├── xgb_weather_model.json    # trained weather-only XGBoost model
├── weather_columns.pkl       # feature order expected by the model
├── extra_info.pkl            # constants + default feature values
├── monitoring_results.csv    # actual vs. predicted + anomaly flags
└── README.md
```

---

## ⚠️ Limitations

- The data contains energy output and weather, but **no recorded equipment faults**. The system detects *deviations from expected behavior*; it does **not** prove a fault in a plant. A flagged hour means "worth reviewing".
- Train/test split is random. A strictly chronological split would be a stricter test of generalization.
- In the dashboard, `isSun` is approximated from the hour (06:00–18:00), and day length / sunlight time use dataset medians.
- The anomaly threshold is a tunable business rule, not a learned one.

## 🔭 Future Work

- Chronological (walk-forward) validation
- Learned anomaly detectors (e.g. Isolation Forest) on residuals
- Seasonal / hour-aware thresholds
- Live weather API integration

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Matplotlib` · `Seaborn` · `Streamlit`

<div align="center">

⭐ If you find this project useful, consider giving it a star.

</div>
