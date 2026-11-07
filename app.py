import numpy as np
import pandas as pd
import joblib
import streamlit as st
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Renewable Energy Guardian", layout="wide")

# ---------- تحميل الملفات ----------
model = XGBRegressor()
model.load_model("xgb_weather_model.json")
columns = joblib.load("weather_columns.pkl")
extra = joblib.load("extra_info.pkl")
monitor = pd.read_csv("monitoring_results.csv", parse_dates=["Time"])

ghi_max = extra["ghi_max"]
ghi_q75 = extra["ghi_q75"]
defaults = extra["defaults"]

# ---------- القائمة الجانبية ----------
st.sidebar.title("Smart Renewable Energy Guardian")
page = st.sidebar.radio(
    "Pages",
    ["Home", "Energy Prediction", "Anomaly Detection", "What-if Simulation", "Model Insights"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Weather conditions")
ghi = st.sidebar.slider("GHI (solar radiation)", 0, 230, 100)
temp = st.sidebar.slider("Temperature (°C)", -15, 36, 10)
clouds = st.sidebar.slider("Clouds (%)", 0, 100, 50)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 80)
wind = st.sidebar.slider("Wind speed (m/s)", 0.0, 15.0, 4.0)
hour = st.sidebar.slider("Hour", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)


# ---------- بناء صف الإدخال للموديل ----------
# base = الظروف الحالية ، new = الظروف الجديدة في What-if
new_ghi = ghi
new_temp = temp
new_clouds = clouds
if page == "What-if Simulation":
    st.sidebar.markdown("---")
    st.sidebar.subheader("New conditions (What-if)")
    new_ghi = st.sidebar.slider("New GHI", 0, 230, min(ghi + 50, 230))
    new_temp = st.sidebar.slider("New Temperature (°C)", -15, 36, temp)
    new_clouds = st.sidebar.slider("New Clouds (%)", 0, 100, max(clouds - 20, 0))

scenarios = {
    "base": [ghi, temp, clouds],
    "new": [new_ghi, new_temp, new_clouds],
}
rows = {}

for name in scenarios:
    g, t, c = scenarios[name]
    r = dict(defaults)
    r["GHI"] = g
    r["temp"] = t
    r["clouds_all"] = c
    r["humidity"] = humidity
    r["wind_speed"] = wind
    r["hour"] = hour
    r["month"] = month
    r["day_of_year"] = month * 30 - 15
    r["isSun"] = 1 if 6 <= hour <= 18 else 0
    r["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    r["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    r["month_sin"] = np.sin(2 * np.pi * month / 12)
    r["month_cos"] = np.cos(2 * np.pi * month / 12)
    r["day_of_year_sin"] = np.sin(2 * np.pi * r["day_of_year"] / 365.25)
    r["day_of_year_cos"] = np.cos(2 * np.pi * r["day_of_year"] / 365.25)
    r["GHI_normalized"] = g / ghi_max
    r["GHI_temp"] = g * t
    r["GHI_humidity"] = g * humidity
    r["clear_sky_indicator"] = int(g > ghi_q75 and c < 30)
    r["cloud_effect"] = 1 - c / 100
    r["wind_temp"] = wind * t
    rows[name] = pd.DataFrame([r])[columns]

pred_base = max(float(model.predict(rows["base"])[0]), 0)
pred_new = max(float(model.predict(rows["new"])[0]), 0)
if rows["base"]["isSun"].values[0] == 0:
    pred_base = 0
    pred_new = 0

# =====================================================
# Home
# =====================================================
if page == "Home":
    st.title("Smart Renewable Energy Guardian")
    st.write("نظام ذكي للتنبؤ بإنتاج الطاقة المتجددة ومراقبة كفاءة الأداء واكتشاف الانحرافات غير الطبيعية.")

    anomalies = monitor[monitor["is_anomaly"] == 1]
    under = monitor[monitor["status"] == "Underperformance"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitored hours", f"{len(monitor):,}")
    c2.metric("Anomalies", f"{len(anomalies):,}")
    c3.metric("Anomaly rate", f"{len(anomalies) / len(monitor) * 100:.2f}%")
    c4.metric("Underperformance", f"{len(under):,}")

    st.subheader("Status distribution")
    counts = monitor["status"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.index, counts.values, color=["#4c78a8", "#e45756", "#f58518"][: len(counts)])
    ax.set_ylabel("Hours")
    st.pyplot(fig)

    st.info("ملاحظة: النظام يكتشف انحرافات عن السلوك المتوقع، ولا يثبت وجود عطل فعلي في المحطة.")

# =====================================================
# Energy Prediction
# =====================================================
elif page == "Energy Prediction":
    st.title("Energy Prediction")
    st.write("غيّر الظروف الجوية من القائمة الجانبية وشوف التوقع.")

    st.metric("Predicted energy", f"{pred_base:,.0f} Wh")

    st.subheader("Compare with actual production")
    actual = st.number_input("Actual production (Wh)", min_value=0, value=0, step=10)
    threshold = st.slider("Anomaly threshold (%)", 20, 150, 70)

    if actual > 0 and pred_base > 100:
        dev = (actual - pred_base) / pred_base * 100
        st.write(f"Deviation: **{dev:.1f}%**")
        if abs(dev) > threshold and dev < 0:
            st.error("Underperformance: الإنتاج الفعلي أقل من المتوقع بشكل ملحوظ، الحالة تحتاج مراجعة.")
        elif abs(dev) > threshold and dev > 0:
            st.warning("Overperformance: الإنتاج الفعلي أعلى من المتوقع بشكل ملحوظ.")
        else:
            st.success("Normal: الإنتاج ضمن النطاق الطبيعي.")
    elif actual > 0:
        st.info("التوقع صغير جدًا (أقل من 100 Wh)، فنسبة الانحراف مش مفيدة هنا.")

# =====================================================
# Anomaly Detection
# =====================================================
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")

    months = sorted(monitor["Time"].dt.to_period("M").astype(str).unique())
    chosen = st.selectbox("Month", months)
    sub = monitor[monitor["Time"].dt.to_period("M").astype(str) == chosen]

    st.write(f"Hours: {len(sub)} | Anomalies: {int(sub['is_anomaly'].sum())}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["Time"], sub["actual"], label="Actual", alpha=0.7)
    ax.plot(sub["Time"], sub["predicted"], label="Predicted", alpha=0.7)
    a = sub[sub["is_anomaly"] == 1]
    ax.scatter(a["Time"], a["actual"], color="red", s=25, label="Anomaly", zorder=5)
    ax.legend()
    ax.set_ylabel("Wh")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    n = monitor[monitor["is_anomaly"] == 0].sample(min(4000, len(monitor)), random_state=1)
    a2 = monitor[monitor["is_anomaly"] == 1]
    ax2.scatter(n["predicted"], n["actual"], s=6, alpha=0.3, label="Normal")
    ax2.scatter(a2["predicted"], a2["actual"], s=8, color="red", label="Anomaly")
    ax2.plot([0, 5000], [0, 5000], "k--")
    ax2.set_xlabel("Predicted (Wh)")
    ax2.set_ylabel("Actual (Wh)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Top 10 underperformance cases")
    worst = monitor[monitor["status"] == "Underperformance"].sort_values("difference").head(10)
    st.dataframe(worst[["Time", "actual", "predicted", "deviation_pct"]])

# =====================================================
# What-if Simulation
# =====================================================
elif page == "What-if Simulation":
    st.title("What-if Simulation")
    st.write("غيّر الظروف الجديدة من القائمة الجانبية وقارن التأثير على التوقع.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Before", f"{pred_base:,.0f} Wh")
    c2.metric("After", f"{pred_new:,.0f} Wh")
    c3.metric("Change", f"{pred_new - pred_base:,.0f} Wh")

    # تأثير GHI
    ghi_values = list(range(0, 231, 10))
    preds_ghi = []
    for g in ghi_values:
        row = rows["base"].copy()
        row["GHI"] = g
        row["GHI_normalized"] = g / ghi_max
        row["GHI_temp"] = g * temp
        row["GHI_humidity"] = g * humidity
        row["clear_sky_indicator"] = int(g > ghi_q75 and clouds < 30)
        preds_ghi.append(max(float(model.predict(row)[0]), 0))

    # تأثير السحب
    cloud_values = list(range(0, 101, 10))
    preds_cloud = []
    for c in cloud_values:
        row = rows["base"].copy()
        row["clouds_all"] = c
        row["cloud_effect"] = 1 - c / 100
        row["clear_sky_indicator"] = int(ghi > ghi_q75 and c < 30)
        preds_cloud.append(max(float(model.predict(row)[0]), 0))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ghi_values, preds_ghi, marker="o")
    ax[0].set_title("Effect of GHI")
    ax[0].set_xlabel("GHI")
    ax[0].set_ylabel("Predicted Energy (Wh)")
    ax[0].grid(True)
    ax[1].plot(cloud_values, preds_cloud, marker="o", color="orange")
    ax[1].set_title("Effect of Clouds")
    ax[1].set_xlabel("Clouds (%)")
    ax[1].grid(True)
    st.pyplot(fig)

# =====================================================
# Model Insights
# =====================================================
elif page == "Model Insights":
    st.title("Model Insights")

    st.subheader("Model performance (test set)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", "0.937")
    c2.metric("MAE", "110 Wh")
    c3.metric("RMSE", "262 Wh")

    st.subheader("Feature importance")
    imp = pd.Series(model.feature_importances_, index=columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values)
    st.pyplot(fig)

    st.subheader("Why this prediction? (SHAP)")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(rows["base"])
        exp = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=rows["base"].iloc[0],
            feature_names=columns,
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning("SHAP غير متاح. ثبّته بالأمر: pip install shap")
