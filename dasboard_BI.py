import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ | Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── MODERN BI CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 1. Change the global background */
    .stApp {
        background-color: #0B0E14; 
    }

    /* 2. Style the containers to look like 'Tiles' */
    [data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #161B22 !important; /* Card Color */
        border: 1px solid #30363D !important; /* Subtle Border */
        border-radius: 8px !important;
        padding: 20px !important;
    }

    /* 3. Customize the Sidebar to match */
    section[data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363D !important;
    }

    /* 4. Hide the "Made with Streamlit" footer and header */
    #MainMenu, footer, header { visibility: hidden; }

    /* 5. Custom Font (Inter or Montserrat) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── DATA ENGINE ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    return df

try:
    df_raw = load_data()
except:
    st.error("Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the folder.")
    st.stop()

# ─── SIDEBAR FILTERS (POWER BI SLICERS) ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Global Slicers")
    contract = st.multiselect("Contract Type", df_raw['Contract'].unique(), default=df_raw['Contract'].unique())
    internet = st.multiselect("Internet Service", df_raw['InternetService'].unique(), default=df_raw['InternetService'].unique())
    payment = st.multiselect("Payment Method", df_raw['PaymentMethod'].unique(), default=df_raw['PaymentMethod'].unique())
    
    st.divider()
    st.caption("B.Tech IT Final Year Project\nPortfolio Build v2.0")

# Apply Filters
df = df_raw[
    (df_raw['Contract'].isin(contract)) & 
    (df_raw['InternetService'].isin(internet)) & 
    (df_raw['PaymentMethod'].isin(payment))
]

# ─── DASHBOARD TOP ROW: KPIs ──────────────────────────────────────────────────
st.markdown('<div class="bi-header">CHURN ANALYTICS EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    with st.container(border=True):
        churn_rate = (df['Churn_bin'].mean() * 100)
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%", f"{df['Churn_bin'].sum()} Users")

with kpi2:
    with st.container(border=True):
        revenue_at_risk = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
        st.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", "High Risk", delta_color="inverse")

with kpi3:
    with st.container(border=True):
        avg_tenure = df['tenure'].mean()
        st.metric("Avg. Customer Tenure", f"{avg_tenure:.1f} mo")

with kpi4:
    with st.container(border=True):
        high_val_churn = df[(df['Churn']=='Yes') & (df['MonthlyCharges'] > 80)].shape[0]
        st.metric("High-Value Churners", high_val_churn, "Focus Area")

st.markdown("<br>", unsafe_allow_html=True)

# ─── MIDDLE ROW: GRID LAYOUT ──────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    with st.container(border=True):
        st.markdown("**Churn by Contract & Service**")
        # Power BI Style Stacked Bar
        fig_bar = px.histogram(
            df, x="Contract", color="Churn", 
            barmode="group",
            color_discrete_map={"Yes": "#f85149", "No": "#238636"},
            template="plotly_dark", height=350
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    with st.container(border=True):
        st.markdown("**Risk Distribution (Monthly Charges)**")
        # Power BI Style Density Plot
        fig_kde = px.box(
            df, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map={"Yes": "#f85149", "No": "#238636"},
            points="all", template="plotly_dark", height=350
        )
        fig_kde.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_kde, use_container_width=True)

# ─── BOTTOM ROW: DEEP DIVE ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("**Churn Probability Factor Analysis**")
    
    # Simple heatmap/correlation proxy
    corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_bin']
    corr_matrix = df[corr_cols].corr()
    
    fig_heat = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdYlGn_r',
        template="plotly_dark",
        height=300
    )
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)

# ─── DATA PREVIEW (DRILL THROUGH) ─────────────────────────────────────────────
with st.expander("📂 Raw Data Drill-Through (Filtered Records)"):
    st.dataframe(df, use_container_width=True)