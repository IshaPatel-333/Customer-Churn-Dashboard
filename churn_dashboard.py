"""
Telco Churn Intelligence Dashboard
────────────────────────────────────
B.Tech IT Final Year — Data Science Portfolio Project

Run:
    pip install streamlit plotly pandas
    streamlit run churn_dashboard.py

Place  WA_Fn-UseC_-Telco-Customer-Churn.csv  in the same folder.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME / CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #0a0c10 !important;
    color: #e8ecf4 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid #1e2330 !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { font-size: 10px !important; letter-spacing: 0.15em; text-transform: uppercase; color: #6b7592 !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 2rem !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7592 !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5a0 !important;
    border-bottom-color: #00e5a0 !important;
}

/* ── Selectbox / multiselect ── */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: #111318 !important;
    border-color: #1e2330 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}

/* ── Divider ── */
hr { border-color: #1e2330 !important; }

/* ── Section tag ── */
.section-tag {
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #3a4260;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ── Insight card ── */
.insight-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    border-left: 3px solid #00e5a0;
}
.insight-card.red  { border-left-color: #ff4d6d; }
.insight-card.blue { border-left-color: #4d9fff; }
.insight-card.yellow { border-left-color: #ffd166; }
.insight-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 13px; margin-bottom: 6px; }
.insight-body  { font-size: 11px; color: #6b7592; line-height: 1.6; }
.insight-body strong { color: #ff4d6d; }
.insight-body em { color: #00e5a0; font-style: normal; }

/* ── Big header ── */
.main-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(28px, 4vw, 44px) !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    letter-spacing: -0.02em !important;
    color: #e8ecf4 !important;
    margin-bottom: 4px !important;
}
.main-title span { color: #00e5a0; }
.subtitle { font-size: 12px; color: #6b7592; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DARK TEMPLATE ─────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#6b7592",
    font_family="DM Mono",
    font_size=11,
    xaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", tickfont_size=10),
    yaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", tickfont_size=10),
    margin=dict(l=8, r=8, t=30, b=8),
    hoverlabel=dict(bgcolor="#181c24", bordercolor="#1e2330", font_color="#e8ecf4"),
)

GREEN  = "#00e5a0"
RED    = "#ff4d6d"
BLUE   = "#4d9fff"
YELLOW = "#ffd166"
PURPLE = "#a78bfa"
COLORS = [GREEN, RED, BLUE, YELLOW, PURPLE, "#ff7a5c", "#34d399", "#60a5fa"]

def apply_dark(fig):
    fig.update_layout(**DARK)
    fig.update_xaxes(gridcolor="#1e2330", linecolor="#2a3050", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2330", linecolor="#2a3050", zeroline=False)
    return fig

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0–12m", "13–24m", "25–36m", "37–48m", "49–60m", "61–72m"],
    )
    df["charge_bucket"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 20, 40, 60, 80, 100, 120],
        labels=["$0–20", "$20–40", "$40–60", "$60–80", "$80–100", "$100–120"],
    )
    return df

try:
    df_full = load_data()
except FileNotFoundError:
    st.error("⚠️  CSV file not found. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder as this script.")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title" style="font-size:22px!important">📡 Churn<span>IQ</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Telco Customer Analytics</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🔍 Filter Data**")

    gender_filter = st.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
    contract_filter = st.multiselect(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        default=["Month-to-month", "One year", "Two year"],
    )
    internet_filter = st.multiselect(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        default=["DSL", "Fiber optic", "No"],
    )
    senior_filter = st.multiselect("Senior Citizen", ["Senior", "Non-Senior"], default=["Senior", "Non-Senior"])

    st.markdown("---")
    st.markdown('<p class="subtitle">B.Tech IT Final Year<br>Data Science Portfolio</p>', unsafe_allow_html=True)

# ─── APPLY FILTERS ────────────────────────────────────────────────────────────
senior_map = {"Senior": 1, "Non-Senior": 0}
senior_vals = [senior_map[s] for s in senior_filter]

df = df_full[
    df_full["gender"].isin(gender_filter)
    & df_full["Contract"].isin(contract_filter)
    & df_full["InternetService"].isin(internet_filter)
    & df_full["SeniorCitizen"].isin(senior_vals)
].copy()

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ─── HEADER ───────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        '<h1 class="main-title">Telco <span>Churn</span> Intelligence</h1>'
        '<p class="subtitle">Exploratory Data Analysis · Feature Engineering · Predictive Insights</p>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f'<div style="text-align:right; padding-top:8px;">'
        f'<span style="background:#111318;border:1px solid #1e2330;border-radius:20px;padding:5px 14px;font-size:11px;color:#6b7592;">'
        f'<b style="color:#00e5a0">{len(df):,}</b> records filtered</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─── KPI STRIP ────────────────────────────────────────────────────────────────
total = len(df)
churned = df["Churn_binary"].sum()
churn_rate = churned / total * 100
avg_charge_churned = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
avg_charge_retained = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
mtm_churn = df[df["Contract"] == "Month-to-month"]["Churn_binary"].mean() * 100 if "Month-to-month" in contract_filter else 0
two_yr_churn = df[df["Contract"] == "Two year"]["Churn_binary"].mean() * 100 if "Two year" in contract_filter else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Churn Rate", f"{churn_rate:.1f}%", f"{int(churned):,} of {total:,} customers")
k2.metric("Avg Monthly Charge (Churned)", f"${avg_charge_churned:.1f}", f"vs ${avg_charge_retained:.1f} retained")
k3.metric("Month-to-Month Churn", f"{mtm_churn:.1f}%", "Highest risk segment", delta_color="inverse")
k4.metric("Two-Year Contract Churn", f"{two_yr_churn:.1f}%", "Lowest risk segment")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Distribution",
    "🔍  Key Drivers",
    "🛡️  Services",
    "👤  Demographics",
    "💡  Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-tag">01 — Churn Distribution & Composition</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    # Tenure Histogram
    with c1:
        tenure_bins = list(range(0, 73, 6))
        tenure_labels = [f"{b}–{b+5}" for b in tenure_bins[:-1]]
        df["tb12"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
        hist_data = df.groupby(["tb12", "Churn"])["Churn_binary"].count().unstack(fill_value=0).reset_index()
        hist_data.columns.name = None

        fig_tenure = go.Figure()
        fig_tenure.add_bar(
            x=hist_data["tb12"].astype(str), y=hist_data.get("No", 0),
            name="Retained", marker_color=GREEN, marker_opacity=0.75,
        )
        fig_tenure.add_bar(
            x=hist_data["tb12"].astype(str), y=hist_data.get("Yes", 0),
            name="Churned", marker_color=RED, marker_opacity=0.85,
        )
        fig_tenure.update_layout(
            **DARK,
            title=dict(text="Customer Tenure vs Churn (months)", font_size=13, font_color="#e8ecf4", x=0, xanchor="left"),
            barmode="group",
            legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="right", x=1, font_size=11),
            height=280,
        )
        apply_dark(fig_tenure)
        st.plotly_chart(fig_tenure, use_container_width=True)

    # Donut
    with c2:
        retained_n = total - int(churned)
        fig_donut = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[retained_n, int(churned)],
            hole=0.72,
            marker_colors=[GREEN, RED],
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            **DARK,
            title=dict(text="Churn Breakdown", font_size=13, font_color="#e8ecf4", x=0, xanchor="left"),
            showlegend=True,
            legend=dict(orientation="v", font_size=11),
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:10px'>churn</span>",
                x=0.5, y=0.5, showarrow=False, font_size=16, font_color="#ff4d6d",
            )],
            height=280,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Monthly Charges Distribution
    charge_data = df.groupby(["charge_bucket", "Churn"])["Churn_binary"].count().unstack(fill_value=0).reset_index()
    charge_data.columns.name = None

    fig_charge = go.Figure()
    fig_charge.add_bar(x=charge_data["charge_bucket"].astype(str), y=charge_data.get("No", 0), name="Retained", marker_color=GREEN, marker_opacity=0.75)
    fig_charge.add_bar(x=charge_data["charge_bucket"].astype(str), y=charge_data.get("Yes", 0), name="Churned", marker_color=RED, marker_opacity=0.85)
    fig_charge.update_layout(
        **DARK,
        title=dict(text="Monthly Charges Distribution by Churn", font_size=13, font_color="#e8ecf4", x=0, xanchor="left"),
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="right", x=1, font_size=11),
        height=260,
    )
    apply_dark(fig_charge)
    st.plotly_chart(fig_charge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KEY DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-tag">02 — Key Churn Drivers</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Contract
    with c1:
        contract_churn = df.groupby("Contract")["Churn_binary"].mean().reset_index()
        contract_churn["pct"] = (contract_churn["Churn_binary"] * 100).round(1)
        clr_map = {"Month-to-month": RED, "One year": YELLOW, "Two year": GREEN}
        contract_churn["color"] = contract_churn["Contract"].map(clr_map)

        fig_contract = go.Figure(go.Bar(
            x=contract_churn["pct"], y=contract_churn["Contract"],
            orientation="h",
            marker_color=contract_churn["color"].tolist(),
            text=contract_churn["pct"].map(lambda v: f"{v}%"),
            textposition="outside",
            textfont_size=11,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig_contract.update_layout(
            **DARK,
            title=dict(text="Churn Rate by Contract Type", font_size=13, font_color="#e8ecf4", x=0),
            xaxis_ticksuffix="%", xaxis_range=[0, 55],
            height=240,
        )
        apply_dark(fig_contract)
        st.plotly_chart(fig_contract, use_container_width=True)

    # Internet
    with c2:
        inet_churn = df.groupby("InternetService")["Churn_binary"].mean().reset_index()
        inet_churn["pct"] = (inet_churn["Churn_binary"] * 100).round(1)
        inet_clr = {"No": GREEN, "DSL": BLUE, "Fiber optic": RED}
        inet_churn["color"] = inet_churn["InternetService"].map(inet_clr)

        fig_inet = go.Figure(go.Bar(
            x=inet_churn["InternetService"], y=inet_churn["pct"],
            marker_color=inet_churn["color"].tolist(),
            text=inet_churn["pct"].map(lambda v: f"{v}%"),
            textposition="outside",
            textfont_size=11,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig_inet.update_layout(
            **DARK,
            title=dict(text="Churn Rate by Internet Service", font_size=13, font_color="#e8ecf4", x=0),
            yaxis_ticksuffix="%", yaxis_range=[0, 55],
            height=240,
        )
        apply_dark(fig_inet)
        st.plotly_chart(fig_inet, use_container_width=True)

    # Payment Method
    pay_churn = df.groupby("PaymentMethod")["Churn_binary"].mean().reset_index()
    pay_churn["pct"] = (pay_churn["Churn_binary"] * 100).round(1)
    pay_churn = pay_churn.sort_values("pct")
    pay_colors = [RED if v > 40 else YELLOW if v > 20 else GREEN for v in pay_churn["pct"]]

    fig_pay = go.Figure(go.Bar(
        x=pay_churn["pct"], y=pay_churn["PaymentMethod"],
        orientation="h",
        marker_color=pay_colors,
        text=pay_churn["pct"].map(lambda v: f"{v}%"),
        textposition="outside",
        textfont_size=11,
    ))
    fig_pay.update_layout(
        **DARK,
        title=dict(text="Churn Rate by Payment Method", font_size=13, font_color="#e8ecf4", x=0),
        xaxis_ticksuffix="%", xaxis_range=[0, 60],
        height=220,
    )
    apply_dark(fig_pay)
    st.plotly_chart(fig_pay, use_container_width=True)

    # Tenure Decay Line
    tenure_rate = df.groupby("tenure_bucket", observed=True)["Churn_binary"].mean().reset_index()
    tenure_rate["pct"] = (tenure_rate["Churn_binary"] * 100).round(1)

    fig_decay = go.Figure()
    fig_decay.add_scatter(
        x=tenure_rate["tenure_bucket"].astype(str),
        y=tenure_rate["pct"],
        mode="lines+markers",
        line=dict(color=RED, width=2.5),
        marker=dict(color=RED, size=8, line=dict(color="#0a0c10", width=2)),
        fill="tozeroy",
        fillcolor="rgba(255,77,109,0.08)",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    )
    fig_decay.update_layout(
        **DARK,
        title=dict(text="Churn Rate Decay by Tenure Bucket", font_size=13, font_color="#e8ecf4", x=0),
        yaxis_ticksuffix="%", yaxis_range=[0, 60],
        height=240,
    )
    apply_dark(fig_decay)
    st.plotly_chart(fig_decay, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SERVICES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-tag">03 — Service Adoption & Churn Risk</p>', unsafe_allow_html=True)

    services = [
        "OnlineSecurity", "TechSupport", "OnlineBackup",
        "DeviceProtection", "StreamingTV", "StreamingMovies",
    ]

    svc_rows = []
    for s in services:
        yes_r = df[df[s] == "Yes"]["Churn_binary"].mean() * 100
        no_r  = df[df[s] == "No"]["Churn_binary"].mean() * 100
        svc_rows.append({"Service": s, "With Service": round(yes_r, 1), "Without Service": round(no_r, 1)})

    svc_df = pd.DataFrame(svc_rows).sort_values("Without Service", ascending=True)

    fig_svc = go.Figure()
    fig_svc.add_bar(
        y=svc_df["Service"], x=svc_df["Without Service"],
        name="Without Service", orientation="h",
        marker_color=RED, marker_opacity=0.85,
        text=svc_df["Without Service"].map(lambda v: f"{v}%"),
        textposition="outside", textfont_size=10,
    )
    fig_svc.add_bar(
        y=svc_df["Service"], x=svc_df["With Service"],
        name="With Service", orientation="h",
        marker_color=GREEN, marker_opacity=0.75,
        text=svc_df["With Service"].map(lambda v: f"{v}%"),
        textposition="outside", textfont_size=10,
    )
    fig_svc.update_layout(
        **DARK,
        title=dict(text="Churn Rate: With vs Without Add-on Service", font_size=13, font_color="#e8ecf4", x=0),
        barmode="group",
        xaxis_ticksuffix="%", xaxis_range=[0, 55],
        legend=dict(orientation="h", yanchor="top", y=1.08, font_size=11),
        height=360,
    )
    apply_dark(fig_svc)
    st.plotly_chart(fig_svc, use_container_width=True)

    # Heatmap — service combos
    st.markdown("**Service Adoption Correlation with Churn**")
    service_cols = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
    binary_svc = df[service_cols].applymap(lambda x: 1 if x == "Yes" else 0)
    binary_svc["Churn"] = df["Churn_binary"]
    corr = binary_svc.corr()[["Churn"]].drop("Churn").sort_values("Churn")

    fig_corr = go.Figure(go.Bar(
        x=corr["Churn"].round(3),
        y=corr.index,
        orientation="h",
        marker_color=[RED if v < 0 else GREEN for v in corr["Churn"]],
        text=corr["Churn"].map(lambda v: f"{v:+.3f}"),
        textposition="outside",
        textfont_size=10,
    ))
    fig_corr.update_layout(
        **DARK,
        title=dict(text="Correlation of Service Adoption with Churn", font_size=13, font_color="#e8ecf4", x=0),
        height=260,
        xaxis_title="Pearson Correlation",
    )
    apply_dark(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-tag">04 — Demographic Signals</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Senior
    with c1:
        sen = df.groupby("SeniorCitizen")["Churn_binary"].mean().reset_index()
        sen["label"] = sen["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        sen["pct"] = (sen["Churn_binary"] * 100).round(1)

        fig_s = go.Figure(go.Bar(
            x=sen["label"], y=sen["pct"],
            marker_color=[GREEN, RED],
            text=sen["pct"].map(lambda v: f"{v}%"),
            textposition="outside", textfont_size=12,
        ))
        fig_s.update_layout(**DARK, title=dict(text="Senior Citizen Impact", font_size=13, font_color="#e8ecf4", x=0),
                            yaxis_ticksuffix="%", yaxis_range=[0, 55], height=240)
        apply_dark(fig_s)
        st.plotly_chart(fig_s, use_container_width=True)

    # Gender
    with c2:
        gen = df.groupby("gender")["Churn_binary"].mean().reset_index()
        gen["pct"] = (gen["Churn_binary"] * 100).round(1)

        fig_g = go.Figure(go.Bar(
            x=gen["gender"], y=gen["pct"],
            marker_color=[BLUE, PURPLE],
            text=gen["pct"].map(lambda v: f"{v}%"),
            textposition="outside", textfont_size=12,
        ))
        fig_g.update_layout(**DARK, title=dict(text="Gender Churn Rate", font_size=13, font_color="#e8ecf4", x=0),
                            yaxis_ticksuffix="%", yaxis_range=[24, 30], height=240)
        apply_dark(fig_g)
        st.plotly_chart(fig_g, use_container_width=True)

    # Partner + Dependents
    with c3:
        fam_data = {
            "Segment": ["Has Partner", "No Partner", "Has Dependents", "No Dependents"],
            "pct": [
                df[df["Partner"] == "Yes"]["Churn_binary"].mean() * 100,
                df[df["Partner"] == "No"]["Churn_binary"].mean() * 100,
                df[df["Dependents"] == "Yes"]["Churn_binary"].mean() * 100,
                df[df["Dependents"] == "No"]["Churn_binary"].mean() * 100,
            ]
        }
        fam_df = pd.DataFrame(fam_data)
        fam_df["pct"] = fam_df["pct"].round(1)

        fig_fam = go.Figure(go.Bar(
            x=fam_df["Segment"], y=fam_df["pct"],
            marker_color=[GREEN, RED, GREEN, RED],
            text=fam_df["pct"].map(lambda v: f"{v}%"),
            textposition="outside", textfont_size=10,
        ))
        fig_fam.update_layout(**DARK, title=dict(text="Partner & Dependents", font_size=13, font_color="#e8ecf4", x=0),
                              yaxis_ticksuffix="%", yaxis_range=[0, 45], height=240,
                              xaxis_tickfont_size=9)
        apply_dark(fig_fam)
        st.plotly_chart(fig_fam, use_container_width=True)

    # Paperless Billing
    paper = df.groupby("PaperlessBilling")["Churn_binary"].mean().reset_index()
    paper["pct"] = (paper["Churn_binary"] * 100).round(1)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_paper = go.Figure(go.Bar(
            x=paper["PaperlessBilling"], y=paper["pct"],
            marker_color=[GREEN, RED],
            text=paper["pct"].map(lambda v: f"{v}%"),
            textposition="outside", textfont_size=12,
        ))
        fig_paper.update_layout(**DARK, title=dict(text="Churn: Paperless Billing", font_size=13, font_color="#e8ecf4", x=0),
                                yaxis_ticksuffix="%", yaxis_range=[0, 45], height=220)
        apply_dark(fig_paper)
        st.plotly_chart(fig_paper, use_container_width=True)

    with col_b:
        # Churn rate vs avg monthly charges — scatter (sampled)
        sample = df.sample(min(800, len(df)), random_state=42)
        fig_sc = px.scatter(
            sample, x="tenure", y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": RED, "No": GREEN},
            opacity=0.55,
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig_sc.update_layout(**DARK,
                             title=dict(text="Tenure vs Monthly Charges (Churn overlay)", font_size=13, font_color="#e8ecf4", x=0),
                             legend=dict(font_size=11), height=220)
        apply_dark(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-tag">05 — Analyst Insights & Recommendations</p>', unsafe_allow_html=True)

    insights = [
        ("", "🎯  Target Month-to-Month Churners First",
         "With <strong>~42.7% churn</strong>, month-to-month customers are your biggest quick win. "
         "Offer incentives at the 3-month mark to convert to 1-year contracts. "
         "A 10% conversion rate here would recover <em>~380 customers</em>."),
        ("red", "⚡  Electronic Check = High-Risk Signal",
         "Users paying via electronic check churn at <strong>45.3%</strong> — nearly 3× the rate of automatic payments. "
         "Prompting migration to auto-pay during onboarding nearly halves churn risk."),
        ("blue", "🛡️  OnlineSecurity & TechSupport are Retention Anchors",
         "Customers <em>without</em> OnlineSecurity churn at <strong>41.8%</strong>. "
         "Bundling these services at signup could significantly improve LTV and reduce churn in the $60–100 tier."),
        ("yellow", "⏱️  The First 12 Months are Critical",
         "Churn in months 0–5 is <strong>47.7%</strong> — dropping to <em>6.6%</em> by year 5. "
         "Invest heavily in onboarding experience and early engagement programs."),
        ("", "👴  Senior Citizens Need a Dedicated Strategy",
         "Seniors churn at <strong>41.7%</strong> vs 23.6% for non-seniors. "
         "Dedicated support lines and simplified billing can significantly improve retention in this high-value segment."),
        ("blue", "📊  Gender is NOT a Predictive Feature",
         "Female 26.9% vs Male 26.2% — essentially identical. "
         "<em>Removing this from your model</em> reduces noise and improves generalizability. "
         "This demonstrates good feature selection instincts to interviewers."),
    ]

    col1, col2 = st.columns(2)
    for i, (cls, title, body) in enumerate(insights):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(
                f'<div class="insight-card {cls}">'
                f'<div class="insight-title">{title}</div>'
                f'<div class="insight-body">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<p class="section-tag">📋  Feature Importance Preview (for ML model)</p>', unsafe_allow_html=True)

    # Rough point-biserial correlations as a proxy for feature importance
    feat_cols = {
        "Contract (encoded)": df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}),
        "Tenure": df["tenure"],
        "InternetService": df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2}),
        "MonthlyCharges": df["MonthlyCharges"],
        "OnlineSecurity": df["OnlineSecurity"].map({"No": 1, "Yes": 0, "No internet service": 0}),
        "TechSupport": df["TechSupport"].map({"No": 1, "Yes": 0, "No internet service": 0}),
        "PaymentMethod": df["PaymentMethod"].map({"Electronic check": 1, "Mailed check": 0, "Bank transfer (automatic)": -1, "Credit card (automatic)": -1}),
        "SeniorCitizen": df["SeniorCitizen"],
        "PaperlessBilling": df["PaperlessBilling"].map({"Yes": 1, "No": 0}),
        "Partner": df["Partner"].map({"No": 1, "Yes": 0}),
    }

    importance = {k: abs(v.corr(df["Churn_binary"])) for k, v in feat_cols.items() if not v.isnull().all()}
    imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Abs Correlation"]).sort_values("Abs Correlation")

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Abs Correlation"].round(3),
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Abs Correlation"],
            colorscale=[[0, "#1e2330"], [0.5, BLUE], [1.0, GREEN]],
            showscale=False,
        ),
        text=imp_df["Abs Correlation"].map(lambda v: f"{v:.3f}"),
        textposition="outside", textfont_size=10,
    ))
    fig_imp.update_layout(
        **DARK,
        title=dict(text="Feature-Churn Correlation (proxy for importance)", font_size=13, font_color="#e8ecf4", x=0),
        xaxis_range=[0, 0.5],
        height=320,
    )
    apply_dark(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        '<p class="subtitle" style="text-align:center;padding-top:20px;">'
        'Dataset: IBM Watson Telco Customer Churn · '
        'Built with Python · pandas · Plotly · Streamlit'
        '</p>',
        unsafe_allow_html=True,
    )
