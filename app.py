import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor, DMatrix

st.set_page_config(
    page_title="Smart Renewable Energy Guardian",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# الألوان والستايل
# =====================================================
AMBER = "#FBBF24"
ORANGE = "#F97316"
TEAL = "#2DD4BF"
BLUE = "#60A5FA"
RED = "#F87171"
GREEN = "#34D399"
MUTED = "#8B95B3"
CARD = "#131A2E"
BORDER = "rgba(255,255,255,0.08)"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family: 'Inter','Cairo',sans-serif; }}

/* خلفية بتدرج خفيف */
.stApp {{
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(251,191,36,.10), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(45,212,191,.08), transparent 55%),
        #0B1020;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}
#MainMenu, footer {{ visibility: hidden; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #10172B 0%, #0D1325 100%);
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.brand {{ display:flex; align-items:center; gap:.7rem; margin-bottom:.4rem; }}
.brand .logo {{
    width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:1.4rem;
    background: linear-gradient(135deg,{AMBER},{ORANGE}); box-shadow:0 6px 20px rgba(249,115,22,.35);
}}
.brand .name {{ font-weight:800; line-height:1.1; font-size:1rem; }}
.brand .sub {{ color:{MUTED}; font-size:.72rem; font-weight:500; }}
.side-label {{
    color:{MUTED}; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;
    font-weight:700; margin:1.1rem 0 .3rem;
}}

/* Nav radio كأزرار */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.25rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    padding:.55rem .8rem; border-radius:10px; border:1px solid transparent; transition:.15s;
    width:100%;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background:rgba(255,255,255,.05); }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
    background: linear-gradient(90deg, rgba(251,191,36,.18), rgba(251,191,36,.04));
    border-color: rgba(251,191,36,.35);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display:none !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] > div:first-child {{ display:none !important; }}

/* Hero */
.hero {{
    position:relative; overflow:hidden; padding:1.8rem 2rem; border-radius:20px; margin-bottom:1.4rem;
    background: linear-gradient(120deg, rgba(251,191,36,.16), rgba(249,115,22,.10) 45%, rgba(45,212,191,.10));
    border:1px solid rgba(251,191,36,.22);
}}
.hero:after {{
    content:"☀️"; position:absolute; right:1.5rem; top:-.6rem; font-size:7rem; opacity:.13; transform:rotate(-8deg);
}}
.hero h1 {{ margin:0; font-size:2rem; font-weight:800; letter-spacing:-.02em; }}
.hero h1 span {{ background:linear-gradient(90deg,{AMBER},{ORANGE}); -webkit-background-clip:text; color:transparent; }}
.hero p, .page-sub, .note, .pill {{ unicode-bidi:plaintext; text-align:left; }}
.hero p {{ margin:.4rem 0 0; color:#B8C0DA; font-size:.98rem; max-width:720px; }}
.page-title {{ font-size:1.9rem; line-height:1.2; font-weight:800; letter-spacing:-.02em; margin:0; }}
.page-sub {{ color:{MUTED}; margin:.2rem 0 1.2rem; }}

/* KPI cards */
.kpi {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:1.1rem 1.2rem; height:100%;
    transition:.2s; position:relative; overflow:hidden;
}}
.kpi:hover {{ transform:translateY(-2px); border-color:rgba(251,191,36,.35); }}
.kpi:before {{
    content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--accent,{AMBER});
}}
.kpi .label {{ color:{MUTED}; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }}
.kpi .value {{ font-size:1.9rem; font-weight:800; margin:.25rem 0 .1rem; letter-spacing:-.02em; }}
.kpi .value small {{ font-size:.9rem; color:{MUTED}; font-weight:600; margin-left:.25rem; }}
.kpi .hint {{ color:{MUTED}; font-size:.78rem; }}

/* Panels */
.panel-title {{ font-weight:700; font-size:1.02rem; margin:.2rem 0 .1rem; }}
.panel-sub {{ color:{MUTED}; font-size:.82rem; margin-bottom:.4rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{BORDER} !important; border-radius:16px !important; background:{CARD}; }}

/* Status pill */
.pill {{ display:inline-flex; align-items:center; gap:.45rem; padding:.45rem .9rem; border-radius:999px; font-weight:700; font-size:.88rem; }}
.pill.ok {{ background:rgba(52,211,153,.14); color:{GREEN}; border:1px solid rgba(52,211,153,.35); }}
.pill.bad {{ background:rgba(248,113,113,.14); color:{RED}; border:1px solid rgba(248,113,113,.35); }}
.pill.warn {{ background:rgba(251,191,36,.14); color:{AMBER}; border:1px solid rgba(251,191,36,.35); }}
.note {{
    background:rgba(96,165,250,.08); border:1px solid rgba(96,165,250,.25); color:#BFD6FB;
    border-radius:12px; padding:.8rem 1rem; font-size:.9rem;
}}

/* السلايدر دايمًا يسار -> يمين حتى لو المتصفح RTL (علشان الدايرة والخط يمشوا مع بعض) */
[data-testid="stSlider"], [data-testid="stSlider"] * {{ direction: ltr !important; }}
[data-testid="stSlider"] [data-testid="stWidgetLabel"] {{ direction: ltr !important; }}

/* Widgets */
div[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER}; padding:1rem 1.2rem; border-radius:16px; }}
.stDataFrame {{ border-radius:12px; overflow:hidden; }}
.stButton>button, .stDownloadButton>button {{ border-radius:10px; font-weight:600; }}
div[data-baseweb="tab-list"] {{ gap:.4rem; }}
button[data-baseweb="tab"] {{ border-radius:10px 10px 0 0; font-weight:600; }}
hr {{ border-color:{BORDER}; }}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# Helpers
# =====================================================
def kpi(label, value, unit="", hint="", accent=AMBER):
    unit_html = f"<small>{unit}</small>" if unit else ""
    st.markdown(
        f"""<div class="kpi" style="--accent:{accent}">
        <div class="label">{label}</div>
        <div class="value">{value}{unit_html}</div>
        <div class="hint">{hint}</div></div>""",
        unsafe_allow_html=True,
    )


def header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def panel_head(title, sub=""):
    st.markdown(f'<div class="panel-title">{title}</div><div class="panel-sub">{sub}</div>', unsafe_allow_html=True)


def style(fig, height=340, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Cairo, sans-serif", color="#C9D1E8", size=12),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        showlegend=legend,
        hoverlabel=dict(bgcolor="#1B2440", bordercolor="rgba(255,255,255,.15)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False, linecolor="rgba(255,255,255,.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def gauge(value, max_value, title="", color=AMBER, suffix=" Wh"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix=suffix, font=dict(size=38, color="#fff"), valueformat=",.0f"),
            title=dict(text=title, font=dict(size=13, color=MUTED)),
            gauge=dict(
                axis=dict(range=[0, max_value], tickcolor=MUTED, tickfont=dict(size=10, color=MUTED)),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(255,255,255,.04)",
                borderwidth=0,
                steps=[
                    dict(range=[0, max_value * 0.33], color="rgba(255,255,255,.03)"),
                    dict(range=[max_value * 0.33, max_value * 0.66], color="rgba(255,255,255,.05)"),
                    dict(range=[max_value * 0.66, max_value], color="rgba(255,255,255,.07)"),
                ],
            ),
        )
    )
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
    )
    return fig


# =====================================================
# تحميل الملفات
# =====================================================
@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    m = XGBRegressor()
    m.load_model("xgb_weather_model.json")
    cols = joblib.load("weather_columns.pkl")
    ex = joblib.load("extra_info.pkl")
    return m, cols, ex


@st.cache_data(show_spinner=False)
def load_monitor():
    return pd.read_csv("monitoring_results.csv", parse_dates=["Time"])


try:
    model, columns, extra = load_assets()
    monitor = load_monitor()
except FileNotFoundError as e:
    st.error(f"ملف ناقص: {e.filename}. شغّل التطبيق من نفس فولدر المشروع (streamlit run app.py).")
    st.stop()
ghi_max = extra["ghi_max"]
ghi_q75 = extra["ghi_q75"]
defaults = extra["defaults"]
GAUGE_MAX = float(np.ceil(monitor["actual"].quantile(0.995) / 500) * 500)

STATUS_COLORS = {"Normal": TEAL, "Underperformance": RED, "Overperformance": AMBER}

# =====================================================
# القائمة الجانبية
# =====================================================
with st.sidebar:
    st.markdown(
        """<div class="brand"><div class="logo">☀️</div>
        <div><div class="name">Energy Guardian</div><div class="sub">Smart Renewable Monitoring</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
    NAV = {
        "🏠  Home": "Home",
        "⚡  Energy Prediction": "Energy Prediction",
        "🚨  Anomaly Detection": "Anomaly Detection",
        "🧪  What-if Simulation": "What-if Simulation",
        "🧠  Model Insights": "Model Insights",
    }
    page = NAV[st.radio("Pages", list(NAV), label_visibility="collapsed")]

    st.markdown('<div class="side-label">Weather conditions</div>', unsafe_allow_html=True)
    ghi = st.slider("☀️ GHI (solar radiation)", 0, 230, 100)
    temp = st.slider("🌡️ Temperature (°C)", -15, 36, 10)
    clouds = st.slider("☁️ Clouds (%)", 0, 100, 50)
    humidity = st.slider("💧 Humidity (%)", 20, 100, 80)
    wind = st.slider("🍃 Wind speed (m/s)", 0.0, 15.0, 4.0)
    c_a, c_b = st.columns(2)
    hour = c_a.slider("🕒 Hour", 0, 23, 12)
    month = c_b.slider("📅 Month", 1, 12, 6)

    new_ghi, new_temp, new_clouds = ghi, temp, clouds
    if page == "What-if Simulation":
        st.markdown('<div class="side-label">New conditions (What-if)</div>', unsafe_allow_html=True)
        new_ghi = st.slider("New GHI", 0, 230, min(ghi + 50, 230))
        new_temp = st.slider("New Temperature (°C)", -15, 36, temp)
        new_clouds = st.slider("New Clouds (%)", 0, 100, max(clouds - 20, 0))


# =====================================================
# بناء صف الإدخال للموديل
# =====================================================
def build_row(g, t, c):
    hum, hr, mo, w = humidity, hour, month, wind
    r = dict(defaults)
    r["GHI"] = g
    r["temp"] = t
    r["clouds_all"] = c
    r["humidity"] = hum
    r["wind_speed"] = w
    r["hour"] = hr
    r["month"] = mo
    r["day_of_year"] = mo * 30 - 15
    r["isSun"] = 1 if 6 <= hr <= 18 else 0
    r["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    r["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    r["month_sin"] = np.sin(2 * np.pi * mo / 12)
    r["month_cos"] = np.cos(2 * np.pi * mo / 12)
    r["day_of_year_sin"] = np.sin(2 * np.pi * r["day_of_year"] / 365.25)
    r["day_of_year_cos"] = np.cos(2 * np.pi * r["day_of_year"] / 365.25)
    r["GHI_normalized"] = g / ghi_max
    r["GHI_temp"] = g * t
    r["GHI_humidity"] = g * hum
    r["clear_sky_indicator"] = int(g > ghi_q75 and c < 30)
    r["cloud_effect"] = 1 - c / 100
    r["wind_temp"] = w * t
    return r


rows = {
    "base": pd.DataFrame([build_row(ghi, temp, clouds)])[columns],
    "new": pd.DataFrame([build_row(new_ghi, new_temp, new_clouds)])[columns],
}
pred_base = max(float(model.predict(rows["base"])[0]), 0)
pred_new = max(float(model.predict(rows["new"])[0]), 0)
if rows["base"]["isSun"].values[0] == 0:
    pred_base = 0
    pred_new = 0


def predict_many(df):
    return np.clip(model.predict(df[columns]), 0, None)


# =====================================================
# Home
# =====================================================
if page == "Home":
    st.markdown(
        """<div class="hero"><h1>Smart Renewable <span>Energy Guardian</span></h1>
        <p>نظام ذكي للتنبؤ بإنتاج الطاقة المتجددة ومراقبة كفاءة الأداء واكتشاف الانحرافات غير الطبيعية.</p></div>""",
        unsafe_allow_html=True,
    )

    anomalies = monitor[monitor["is_anomaly"] == 1]
    under = monitor[monitor["status"] == "Underperformance"]
    over = monitor[monitor["status"] == "Overperformance"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Monitored hours", f"{len(monitor):,}", hint="Daytime hours evaluated", accent=BLUE)
    with c2:
        kpi("Anomalies", f"{len(anomalies):,}", hint="Flagged by deviation rule", accent=AMBER)
    with c3:
        kpi("Anomaly rate", f"{len(anomalies) / max(len(monitor), 1) * 100:.2f}", "%", hint="Share of all hours", accent=ORANGE)
    with c4:
        kpi("Underperformance", f"{len(under):,}", hint=f"{len(over):,} overperformance", accent=RED)

    st.write("")
    left, right = st.columns([1, 2])
    with left:
        with st.container(border=True):
            panel_head("Status distribution", "Share of each hour type")
            counts = monitor["status"].value_counts().reset_index()
            counts.columns = ["status", "hours"]
            fig = go.Figure(
                go.Pie(
                    labels=counts["status"], values=counts["hours"], hole=0.68, sort=False,
                    marker=dict(colors=[STATUS_COLORS.get(s, BLUE) for s in counts["status"]],
                                line=dict(color=CARD, width=3)),
                    textinfo="none", hovertemplate="%{label}: %{value:,} h<extra></extra>",
                )
            )
            fig.add_annotation(text=f"<b>{len(monitor):,}</b><br><span style='font-size:11px;color:{MUTED}'>hours</span>",
                               showarrow=False, font=dict(size=22))
            style(fig, 320)
            fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top"))
            st.plotly_chart(fig, width="stretch")
    with right:
        with st.container(border=True):
            panel_head("Anomalies per month", "Underperformance vs overperformance over time")
            m = monitor[monitor["is_anomaly"] == 1].copy()
            m["month"] = m["Time"].dt.to_period("M").dt.to_timestamp()
            g = m.groupby(["month", "status"]).size().reset_index(name="count")
            fig = px.bar(g, x="month", y="count", color="status", color_discrete_map=STATUS_COLORS)
            fig.update_layout(bargap=0.25, legend_title_text="")
            fig.update_xaxes(title=""); fig.update_yaxes(title="Hours")
            st.plotly_chart(style(fig, 320), width="stretch")

    st.markdown(
        '<div class="note">ℹ️ ملاحظة: النظام يكتشف انحرافات عن السلوك المتوقع، ولا يثبت وجود عطل فعلي في المحطة.</div>',
        unsafe_allow_html=True,
    )

# =====================================================
# Energy Prediction
# =====================================================
elif page == "Energy Prediction":
    header("⚡ Energy Prediction", "غيّر الظروف الجوية من القائمة الجانبية وشوف التوقع.")

    left, right = st.columns([1.1, 1])
    with left:
        with st.container(border=True):
            panel_head("Predicted energy", "Based on the current weather conditions")
            st.plotly_chart(gauge(pred_base, GAUGE_MAX, "Expected output", AMBER), width="stretch")
            if pred_base == 0:
                st.markdown('<span class="pill warn">🌙 Night hour — no solar production expected</span>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            panel_head("Current inputs", "What the model is seeing right now")
            a, b = st.columns(2)
            a.metric("☀️ GHI", f"{ghi}")
            b.metric("🌡️ Temp", f"{temp}°C")
            a.metric("☁️ Clouds", f"{clouds}%")
            b.metric("💧 Humidity", f"{humidity}%")
            a.metric("🍃 Wind", f"{wind} m/s")
            b.metric(f"🕒 Time · Month {month}", f"{hour:02d}:00")

    st.write("")
    with st.container(border=True):
        panel_head("Compare with actual production", "Check whether real output is within the expected range")
        c1, c2 = st.columns(2)
        actual = c1.number_input("Actual production (Wh)", min_value=0, value=0, step=10)
        threshold = c2.slider("Anomaly threshold (%)", 20, 150, 70)

        if actual > 0 and pred_base > 100:
            dev = (actual - pred_base) / pred_base * 100
            k1, k2, k3 = st.columns(3)
            with k1:
                kpi("Predicted", f"{pred_base:,.0f}", "Wh", accent=AMBER)
            with k2:
                kpi("Actual", f"{actual:,.0f}", "Wh", accent=BLUE)
            with k3:
                kpi("Deviation", f"{dev:+.1f}", "%", hint=f"Threshold ±{threshold}%", accent=RED if abs(dev) > threshold else GREEN)
            st.write("")
            if abs(dev) > threshold and dev < 0:
                st.markdown('<span class="pill bad">🔻 Underperformance — الإنتاج الفعلي أقل من المتوقع بشكل ملحوظ، الحالة تحتاج مراجعة.</span>', unsafe_allow_html=True)
            elif abs(dev) > threshold and dev > 0:
                st.markdown('<span class="pill warn">🔺 Overperformance — الإنتاج الفعلي أعلى من المتوقع بشكل ملحوظ.</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill ok">✅ Normal — الإنتاج ضمن النطاق الطبيعي.</span>', unsafe_allow_html=True)
        elif actual > 0:
            st.info("التوقع صغير جدًا (أقل من 100 Wh)، فنسبة الانحراف مش مفيدة هنا.")

# =====================================================
# Anomaly Detection
# =====================================================
elif page == "Anomaly Detection":
    header("🚨 Anomaly Detection", "الفرق بين الإنتاج الفعلي والمتوقع، والساعات اللي محتاجة مراجعة.")

    months = sorted(monitor["Time"].dt.to_period("M").astype(str).unique())
    chosen = st.selectbox("📅 Month", months, index=len(months) - 1)
    sub = monitor[monitor["Time"].dt.to_period("M").astype(str) == chosen]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Hours in month", f"{len(sub):,}", accent=BLUE)
    with c2:
        kpi("Anomalies", f"{int(sub['is_anomaly'].sum()):,}", accent=RED)
    with c3:
        rate = sub["is_anomaly"].mean() * 100 if len(sub) else 0
        kpi("Anomaly rate", f"{rate:.1f}", "%", accent=ORANGE)

    st.write("")
    t1, t2, t3 = st.tabs(["📈 Timeline", "🎯 Predicted vs Actual", "📋 Worst cases"])

    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["Time"], y=sub["actual"], name="Actual", mode="lines",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=sub["Time"], y=sub["predicted"], name="Predicted", mode="lines",
                                 line=dict(color=AMBER, width=2, dash="dot")))
        a = sub[sub["is_anomaly"] == 1]
        fig.add_trace(go.Scatter(x=a["Time"], y=a["actual"], name="Anomaly", mode="markers",
                                 marker=dict(color=RED, size=9, line=dict(color="#fff", width=1))))
        fig.update_yaxes(title="Wh")
        st.plotly_chart(style(fig, 400), width="stretch")

    with t2:
        n = monitor[monitor["is_anomaly"] == 0].sample(min(4000, int((monitor["is_anomaly"] == 0).sum())), random_state=1)
        a2 = monitor[monitor["is_anomaly"] == 1]
        lim = float(max(monitor["actual"].max(), monitor["predicted"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=n["predicted"], y=n["actual"], mode="markers", name="Normal",
                                   marker=dict(color=TEAL, size=4, opacity=0.35)))
        fig.add_trace(go.Scattergl(x=a2["predicted"], y=a2["actual"], mode="markers", name="Anomaly",
                                   marker=dict(color=RED, size=5, opacity=0.7)))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                                 line=dict(color="#fff", dash="dash", width=1)))
        fig.update_xaxes(title="Predicted (Wh)"); fig.update_yaxes(title="Actual (Wh)")
        st.plotly_chart(style(fig, 460), width="stretch")

    with t3:
        worst = monitor[monitor["status"] == "Underperformance"].sort_values("difference").head(10)
        st.dataframe(
            worst[["Time", "actual", "predicted", "deviation_pct"]],
            width="stretch", hide_index=True,
            column_config={
                "Time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                "actual": st.column_config.NumberColumn("Actual (Wh)", format="%.0f"),
                "predicted": st.column_config.NumberColumn("Predicted (Wh)", format="%.0f"),
                "deviation_pct": st.column_config.NumberColumn("Deviation %", format="%.1f%%"),
            },
        )

# =====================================================
# What-if Simulation
# =====================================================
elif page == "What-if Simulation":
    header("🧪 What-if Simulation", "غيّر الظروف الجديدة من القائمة الجانبية وقارن التأثير على التوقع.")

    if pred_base == 0:
        st.markdown('<span class="pill warn">🌙 Night hour (before 06:00 / after 18:00) — expected production is 0, so the simulation is flat.</span>', unsafe_allow_html=True)
        st.write("")
    diff = pred_new - pred_base
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Before", f"{pred_base:,.0f}", "Wh", hint=f"GHI {ghi} · {temp}°C · {clouds}% clouds", accent=BLUE)
    with c2:
        kpi("After", f"{pred_new:,.0f}", "Wh", hint=f"GHI {new_ghi} · {new_temp}°C · {new_clouds}% clouds", accent=AMBER)
    with c3:
        pct = (diff / pred_base * 100) if pred_base > 0 else 0
        kpi("Change", f"{diff:+,.0f}", "Wh", hint=f"{pct:+.1f}% vs before", accent=GREEN if diff >= 0 else RED)

    st.write("")
    gv = np.arange(0, 231, 10)
    df_g = pd.DataFrame([build_row(int(g), temp, clouds) for g in gv])
    preds_ghi = predict_many(df_g) if pred_base > 0 else np.zeros(len(gv))
    cv = np.arange(0, 101, 10)
    df_c = pd.DataFrame([build_row(ghi, temp, int(c)) for c in cv])
    preds_cloud = predict_many(df_c) if pred_base > 0 else np.zeros(len(cv))

    def one(g, c):
        return float(predict_many(pd.DataFrame([build_row(g, temp, c)]))[0]) if pred_base > 0 else 0.0

    after_ghi_y = one(new_ghi, clouds)
    after_cloud_y = one(ghi, new_clouds)

    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            panel_head("Effect of GHI", "Changing GHI only, other conditions kept as current")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gv, y=preds_ghi, mode="lines+markers", name="Prediction",
                                     line=dict(color=AMBER, width=3, shape="spline"), fill="tozeroy",
                                     fillcolor="rgba(251,191,36,.10)", marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=[ghi], y=[pred_base], mode="markers", name="Before",
                                     marker=dict(color=BLUE, size=14, line=dict(color="#fff", width=2))))
            fig.add_trace(go.Scatter(x=[new_ghi], y=[after_ghi_y], mode="markers", name="After",
                                     marker=dict(color=GREEN, size=14, symbol="diamond", line=dict(color="#fff", width=2))))
            fig.update_xaxes(title="GHI"); fig.update_yaxes(title="Predicted Energy (Wh)")
            st.plotly_chart(style(fig, 360), width="stretch")
    with r:
        with st.container(border=True):
            panel_head("Effect of Clouds", "Changing clouds only, other conditions kept as current")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cv, y=preds_cloud, mode="lines+markers", name="Prediction",
                                     line=dict(color=ORANGE, width=3, shape="spline"), fill="tozeroy",
                                     fillcolor="rgba(249,115,22,.10)", marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=[clouds], y=[pred_base], mode="markers", name="Before",
                                     marker=dict(color=BLUE, size=14, line=dict(color="#fff", width=2))))
            fig.add_trace(go.Scatter(x=[new_clouds], y=[after_cloud_y], mode="markers", name="After",
                                     marker=dict(color=GREEN, size=14, symbol="diamond", line=dict(color="#fff", width=2))))
            fig.update_xaxes(title="Clouds (%)"); fig.update_yaxes(title="Predicted Energy (Wh)")
            st.plotly_chart(style(fig, 360), width="stretch")

# =====================================================
# Model Insights
# =====================================================
elif page == "Model Insights":
    header("🧠 Model Insights", "أداء الموديل، أهم العوامل، وتفسير التوقع الحالي.")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("R²", "0.937", hint="Test set", accent=GREEN)
    with c2:
        kpi("MAE", "110", "Wh", hint="Mean absolute error", accent=BLUE)
    with c3:
        kpi("RMSE", "262", "Wh", hint="Root mean squared error", accent=ORANGE)

    st.write("")
    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            panel_head("Feature importance", "Top 12 features in the model")
            imp = pd.Series(model.feature_importances_, index=columns).sort_values().tail(12)
            fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker=dict(color=imp.values, colorscale=[[0, "#3B4A7A"], [1, AMBER]]),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            st.plotly_chart(style(fig, 440, legend=False), width="stretch")
    with r:
        with st.container(border=True):
            panel_head("Why this prediction?", "Feature contributions (SHAP values) for the current scenario")
            try:
                contrib = model.get_booster().predict(DMatrix(rows["base"]), pred_contribs=True)[0]
                bias = float(contrib[-1])
                s = pd.Series(contrib[:-1], index=columns)
                s = s.reindex(s.abs().sort_values(ascending=False).index).head(10)[::-1]
                fig = go.Figure(go.Bar(
                    x=s.values, y=[f"{k} = {rows['base'][k].iloc[0]:.2f}" for k in s.index], orientation="h",
                    marker=dict(color=[GREEN if v >= 0 else RED for v in s.values]),
                    hovertemplate="%{y}<br>%{x:+.1f} Wh<extra></extra>",
                ))
                fig.update_xaxes(title="Impact on prediction (Wh)")
                st.plotly_chart(style(fig, 440, legend=False), width="stretch")
                st.caption(f"Base value: {bias:,.0f} Wh  →  Prediction: {float(contrib.sum()):,.0f} Wh")
            except Exception as e:
                st.warning(f"تعذر حساب تفسير التوقع: {e}")
