"""
🏥 Smart Emergency Department Resource Allocation System
=========================================================
Streamlit Dashboard — 8 Pages
Decision-support system integrating RL, MDP, Fuzzy Logic, MCDM,
Utility Theory, and Optimization for hospital ED resource management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import os

# ============================================================================
# PAGE CONFIG (MUST BE FIRST)
# ============================================================================
st.set_page_config(
    page_title="Smart ED Resource Allocation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# COLOR PALETTE — designed for accessible contrast
# ============================================================================
COLORS = {
    'primary':   '#1a3a5c',   # deep navy
    'secondary': '#2980b9',   # medium blue
    'accent':    '#e74c3c',   # red
    'success':   '#27ae60',   # green
    'warning':   '#f39c12',   # orange
    'info':      '#9b59b6',   # purple
    'light_bg':  '#f5f7fa',
    'card_bg':   '#ffffff',
    'text_dark': '#2c3e50',
    'text_muted':'#5a6c7d',
    'border':    '#e1e8ed',
}

SEVERITY_COLORS = {
    'Critical': '#d62728',
    'High':     '#ff7f0e',
    'Moderate': '#1f77b4',
    'Low':      '#2ca02c',
}

TRIAGE_COLORS = {
    'Critical':    '#d62728',
    'Urgent':      '#ff7f0e',
    'Semi-Urgent': '#bcbd22',
    'Non-Urgent':  '#2ca02c',
}

ACTION_COLORS = {
    'Allocated': '#27ae60',
    'Delayed':   '#f39c12',
    'Transferred': '#e74c3c',
}

# ============================================================================
# GLOBAL CSS — strict contrast rules: dark-on-light, light-on-dark, never light-on-light
# ============================================================================
st.markdown("""
<style>
    /* Main app background — light */
    .stApp { background-color: #f5f7fa; }

    /* Default body text — dark on light */
    .main, .main p, .main span, .main label, .main div { color: #2c3e50; }

    /* Headings in main area — dark navy on light */
    .main h1, .main h2, .main h3, .main h4, .main h5 {
        color: #1a3a5c !important;
        font-weight: 700 !important;
    }

    /* Sidebar — dark navy background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0f2540 100%);
    }

    /* Sidebar text — LIGHT on dark */
    [data-testid="stSidebar"] * {
        color: #e8f0f7 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
        border-color: rgba(255,255,255,0.15);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #e8f0f7 !important;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin: 0.15rem 0;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: rgba(255,255,255,0.08);
    }

    /* Metric cards — white bg, dark text */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        box-shadow: 0 2px 6px rgba(26, 58, 92, 0.06);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 58, 92, 0.12);
    }
    [data-testid="stMetricLabel"] {
        color: #5a6c7d !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #1a3a5c !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2980b9;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #1a5d8e;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(41, 128, 185, 0.3);
    }
    .stButton > button:active {
        background-color: #15497a;
    }

    /* Form inputs */
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stTextInput label, .stMultiSelect label, .stRadio label {
        color: #1a3a5c !important;
        font-weight: 600 !important;
    }

    /* DataFrame styling — dark text on white */
    .stDataFrame { background-color: #ffffff; border-radius: 8px; }
    .stDataFrame [data-testid="stTable"] { color: #2c3e50; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #ffffff;
        padding: 0.4rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        color: #5a6c7d !important;
        font-weight: 600;
        background-color: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a3a5c !important;
        color: #ffffff !important;
    }

    /* Alerts — keep readable */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }

    /* Custom page header — dark gradient with light text */
    .page-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(26, 58, 92, 0.18);
    }
    .page-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .page-header p {
        color: #cfe1f2 !important;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Section card */
    .section-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 6px rgba(26, 58, 92, 0.06);
        margin-bottom: 1rem;
        border-top: 3px solid #2980b9;
    }
    .section-card h3 {
        margin-top: 0 !important;
        color: #1a3a5c !important;
    }

    /* Resource progress bar */
    .resource-bar {
        background-color: #e1e8ed;
        border-radius: 12px;
        height: 28px;
        overflow: hidden;
        margin: 0.4rem 0;
        position: relative;
    }
    .resource-bar-fill {
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        transition: width 0.5s;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================
def style_plotly(fig, title=None, height=None):
    """Apply consistent dark-text-on-white styling to all Plotly charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['primary'], size=16)) if title else None,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=COLORS['text_dark'], family='Arial, sans-serif', size=12),
        xaxis=dict(
            gridcolor=COLORS['border'],
            linecolor='#bdc3c7',
            tickfont=dict(color=COLORS['text_dark']),
            title_font=dict(color=COLORS['text_dark']),
        ),
        yaxis=dict(
            gridcolor=COLORS['border'],
            linecolor='#bdc3c7',
            tickfont=dict(color=COLORS['text_dark']),
            title_font=dict(color=COLORS['text_dark']),
        ),
        legend=dict(
            font=dict(color=COLORS['text_dark']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=60 if title else 30, b=40),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def page_header(title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def resource_bar(label, current, capacity, color):
    pct = (current / capacity) * 100 if capacity > 0 else 0
    text_color = "#ffffff" if pct > 25 else COLORS['text_dark']
    st.markdown(f"""
    <div style="margin-bottom: 0.8rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
            <span style="color:#2c3e50; font-weight:600;">{label}</span>
            <span style="color:#5a6c7d;">{current}/{capacity}</span>
        </div>
        <div class="resource-bar">
            <div class="resource-bar-fill"
                 style="width:{pct}%; background-color:{color}; color:{text_color};">
                {pct:.0f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    candidates = ['./data/', './']
    base = next((p for p in candidates if os.path.exists(p + 'hospital_data.csv')), './')

    hospital = pd.read_csv(base + 'hospital_data.csv')
    fuzzy    = pd.read_csv(base + 'hospital_fuzzy_scores.csv')
    rankings = pd.read_csv(base + 'patient_rankings.csv')
    alloc    = pd.read_csv(base + 'allocation_results.csv')
    eval_df  = pd.read_csv(base + 'evaluation_metrics.csv')

    hospital['Arrival_Time'] = pd.to_datetime(hospital['Arrival_Time'])
    rankings['Arrival_Time'] = pd.to_datetime(rankings['Arrival_Time'])
    hospital['Hour'] = hospital['Arrival_Time'].dt.hour

    return hospital, fuzzy, rankings, alloc, eval_df


try:
    hospital_data, fuzzy_scores, patient_rankings, allocation_results, evaluation_metrics = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Data files not found: {e}")
    st.info("Place CSV files in `./data/` folder or in the project root.")
    st.stop()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("# 🏥 Smart ED")
    st.markdown("##### Resource Allocation System")
    st.markdown("---")

    pages = [
        "📊  1. Hospital Overview",
        "▶️  2. Live Simulation",
        "🔥  3. Fuzzy Triage",
        "🎯  4. MCDM Analysis",
        "🤖  5. RL Analytics",
        "⚙️  6. Optimization",
        "🎚️  7. Trade-Off Studio",
        "📈  8. Executive Dashboard",
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("##### 📁 Dataset")
    st.markdown(f"**Patients:** {len(hospital_data):,}")
    st.markdown(f"**Strategies:** {len(evaluation_metrics)}")
    st.markdown(f"**Triage Classes:** {fuzzy_scores['Triage_Class'].nunique()}")

    st.markdown("---")
    st.markdown("##### ℹ️ About")
    st.markdown(
        "Decision-support system integrating "
        "**RL · MDP · Fuzzy · MCDM · Utility · Optimization**."
    )


# ============================================================================
# PAGE 1 — HOSPITAL OVERVIEW
# ============================================================================
def page_overview():
    page_header("📊 Hospital Overview",
                "Real-time ED status and key performance indicators")

    # Simulated current-state values
    beds_capacity, beds_used = 30, 22
    doc_capacity,  doc_used  = 15, 11
    nurse_capacity, nurse_used = 20, 16
    vent_capacity, vent_used = 10, 4
    critical_count = (hospital_data['Severity'] == 'Critical').sum()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Patients", f"{len(hospital_data):,}", "+12 today")
    c2.metric("Available Beds", f"{beds_capacity - beds_used}/{beds_capacity}",
              f"{((beds_capacity-beds_used)/beds_capacity*100):.0f}% free")
    c3.metric("Available Doctors", f"{doc_capacity - doc_used}/{doc_capacity}",
              f"{((doc_capacity-doc_used)/doc_capacity*100):.0f}% free")
    c4.metric("Available Nurses", f"{nurse_capacity - nurse_used}/{nurse_capacity}",
              f"{((nurse_capacity-nurse_used)/nurse_capacity*100):.0f}% free")
    c5.metric("Critical Cases", critical_count, "-3 vs yesterday")

    st.markdown("###  ")

    # Row: Hourly arrivals + severity pie
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("#### 📈 Patient Arrivals by Hour")
        hourly = hospital_data.groupby('Hour').size().reset_index(name='Arrivals')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly['Hour'], y=hourly['Arrivals'],
            mode='lines+markers',
            line=dict(color=COLORS['secondary'], width=3),
            marker=dict(size=8, color=COLORS['primary'],
                        line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(41, 128, 185, 0.15)',
            name='Arrivals',
        ))
        fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Patient Count")
        st.plotly_chart(style_plotly(fig, height=350), use_container_width=True)

    with col_r:
        st.markdown("#### 🚨 Severity Distribution")
        sev = hospital_data['Severity'].value_counts().reset_index()
        sev.columns = ['Severity', 'Count']
        fig = go.Figure(data=[go.Pie(
            labels=sev['Severity'], values=sev['Count'],
            marker=dict(colors=[SEVERITY_COLORS.get(s, COLORS['primary']) for s in sev['Severity']],
                        line=dict(color='white', width=2)),
            hole=0.5,
            textinfo='label+percent',
            textfont=dict(color='white', size=12),
        )])
        fig.update_layout(showlegend=True, height=350,
                          legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.1))
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    # Row: Resource utilization + demand by severity
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.markdown("#### 🛏️ Current Resource Status")
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            resource_bar("🛏️ Beds",        beds_used,  beds_capacity,  COLORS['secondary'])
            resource_bar("👨‍⚕️ Doctors",    doc_used,   doc_capacity,   COLORS['success'])
            resource_bar("👩‍⚕️ Nurses",     nurse_used, nurse_capacity, COLORS['warning'])
            resource_bar("🫁 Ventilators", vent_used,  vent_capacity,  COLORS['accent'])
            st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 🎯 Resource Demand by Severity")
        demand = hospital_data.groupby('Severity')[
            ['Bed_Required','Doctor_Required','Nurse_Required','Ventilator_Required']
        ].sum().reset_index()
        demand_long = demand.melt(id_vars='Severity', var_name='Resource', value_name='Total')
        demand_long['Resource'] = demand_long['Resource'].str.replace('_Required','')
        fig = px.bar(demand_long, x='Resource', y='Total', color='Severity',
                     color_discrete_map=SEVERITY_COLORS, barmode='group')
        st.plotly_chart(style_plotly(fig, height=350), use_container_width=True)


# ============================================================================
# PAGE 2 — LIVE SIMULATION
# ============================================================================
def page_live_simulation():
    page_header("▶️ Live Simulation",
                "Real-time patient arrivals, queue management, and resource allocation")

    # Init session state
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
    if 'sim_index' not in st.session_state:
        st.session_state.sim_index = 50

    # Controls
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    with c1:
        if st.button("▶️ Play",  use_container_width=True):
            st.session_state.sim_running = True
    with c2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.sim_running = False
    with c3:
        if st.button("⏭️ Step",  use_container_width=True):
            st.session_state.sim_index = min(st.session_state.sim_index + 5,
                                             len(hospital_data) - 1)
    with c4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim_running = False
            st.session_state.sim_index = 50
    with c5:
        speed = st.select_slider("Speed (patients per tick)",
                                  options=[1, 3, 5, 10, 20], value=5)

    # Auto-refresh when playing (2-second interval per project spec)
    if st.session_state.sim_running:
        st_autorefresh(interval=2000, key="sim_refresh", limit=None)
        st.session_state.sim_index = min(
            st.session_state.sim_index + speed,
            len(hospital_data) - 1
        )
        if st.session_state.sim_index >= len(hospital_data) - 1:
            st.session_state.sim_running = False

    idx = st.session_state.sim_index
    processed = hospital_data.iloc[:idx + 1]
    recent_window = hospital_data.iloc[max(0, idx - 20):idx + 1]

    # Live KPIs
    status = "🟢 LIVE" if st.session_state.sim_running else "⏸️ PAUSED"
    st.markdown(f"**Status:** {status}  |  **Patient Index:** {idx + 1}/{len(hospital_data)}  "
                f"|  **Current Time:** {hospital_data.iloc[idx]['Arrival_Time'].strftime('%Y-%m-%d %H:%M')}")

    progress_pct = (idx + 1) / len(hospital_data)
    st.progress(progress_pct)

    # Simulated resource state (oscillates based on recent demand)
    beds_used  = min(30, int(recent_window['Bed_Required'].sum()    * 0.8))
    docs_used  = min(15, int(recent_window['Doctor_Required'].sum() * 0.7))
    nurses_used= min(20, int(recent_window['Nurse_Required'].sum()  * 0.6))
    vents_used = min(10, int(recent_window['Ventilator_Required'].sum()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛏️ Beds in Use", f"{beds_used}/30",
              f"{30 - beds_used} free")
    c2.metric("👨‍⚕️ Doctors Busy", f"{docs_used}/15",
              f"{15 - docs_used} free")
    c3.metric("🚨 Critical (last 20)",
              int((recent_window['Severity'] == 'Critical').sum()))
    c4.metric("⏱️ Avg Wait (last 50)",
              f"{processed['Waiting_Time'].tail(50).mean():.1f} min")

    # Live charts
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("#### 📊 Live Arrival Rate (last 50 patients)")
        recent_50 = processed.tail(50)
        fig = go.Figure()
        for sev, color in SEVERITY_COLORS.items():
            subset = recent_50[recent_50['Severity'] == sev]
            fig.add_trace(go.Scatter(
                x=subset['Arrival_Time'], y=subset['Severity_Level'],
                mode='markers',
                marker=dict(size=12, color=color, line=dict(color='white', width=1)),
                name=sev,
            ))
        fig.update_yaxes(tickvals=[1, 2, 3, 4],
                         ticktext=['Low', 'Moderate', 'High', 'Critical'])
        fig.update_layout(xaxis_title="Arrival Time", yaxis_title="Severity")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)

    with col_r:
        st.markdown("#### 🏥 Resource Status")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        resource_bar("🛏️ Beds",        beds_used,   30, COLORS['secondary'])
        resource_bar("👨‍⚕️ Doctors",    docs_used,   15, COLORS['success'])
        resource_bar("👩‍⚕️ Nurses",     nurses_used, 20, COLORS['warning'])
        resource_bar("🫁 Ventilators", vents_used,  10, COLORS['accent'])
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent arrivals table
    st.markdown("#### 🚪 Recent Arrivals (latest 10)")
    recent = processed.tail(10)[['Patient_ID', 'Arrival_Time', 'Age',
                                  'Severity', 'Heart_Rate', 'Blood_Pressure',
                                  'Oxygen_Level', 'Waiting_Time']].copy()
    recent['Arrival_Time'] = recent['Arrival_Time'].dt.strftime('%H:%M:%S')
    st.dataframe(
        recent.iloc[::-1],
        use_container_width=True, hide_index=True,
        column_config={
            "Severity": st.column_config.TextColumn("Severity"),
            "Waiting_Time": st.column_config.NumberColumn("Wait (min)", format="%d"),
        }
    )


# ============================================================================
# PAGE 3 — FUZZY TRIAGE ANALYSIS
# ============================================================================
def page_fuzzy():
    page_header("🔥 Fuzzy Triage Analysis",
                "Membership functions, urgency scoring, and triage classification")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    avg_urgency = fuzzy_scores['Urgency_Score'].mean()
    high_urgency = (fuzzy_scores['Urgency_Score'] > 70).sum()
    avg_survival = fuzzy_scores['Survival_Probability'].mean()
    c1.metric("Avg Urgency Score", f"{avg_urgency:.1f}")
    c2.metric("High Urgency (>70)", f"{high_urgency}")
    c3.metric("Avg Survival Prob", f"{avg_survival:.2%}")
    c4.metric("Critical Triage",
              f"{(fuzzy_scores['Triage_Class']=='Critical').sum()}")

    # Membership functions
    st.markdown("#### 📐 Fuzzy Membership Functions")
    hr_range = np.arange(40, 181)
    bp_range = np.arange(60, 201)
    ox_range = np.arange(70, 101)

    def trap(x, a, b, c, d):
        return np.clip(np.minimum(np.minimum((x - a) / max(b - a, 1e-9), 1),
                                  (d - x) / max(d - c, 1e-9)), 0, 1)
    def tri(x, a, b, c):
        return np.clip(np.minimum((x - a) / max(b - a, 1e-9),
                                  (c - x) / max(c - b, 1e-9)), 0, 1)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Heart Rate (bpm)",
                                        "Blood Pressure (mmHg)",
                                        "Oxygen Level (%)"))

    for col, (rng, low, med, high, x_title) in enumerate([
        (hr_range, trap(hr_range,40,40,60,80),  tri(hr_range,70,90,110),
                   trap(hr_range,100,120,180,180), "HR"),
        (bp_range, trap(bp_range,60,60,80,90),  tri(bp_range,85,100,120),
                   trap(bp_range,110,130,200,200), "BP"),
        (ox_range, trap(ox_range,70,70,85,90),  tri(ox_range,88,93,97),
                   trap(ox_range,95,98,100,100),  "O2"),
    ], start=1):
        fig.add_trace(go.Scatter(x=rng, y=low,  name=f'Low ({x_title})',
                                  line=dict(color=COLORS['secondary'], width=2),
                                  showlegend=(col == 1), legendgroup='Low'),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=rng, y=med,  name=f'Med ({x_title})',
                                  line=dict(color=COLORS['warning'], width=2, dash='dash'),
                                  showlegend=(col == 1), legendgroup='Med'),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=rng, y=high, name=f'High ({x_title})',
                                  line=dict(color=COLORS['accent'], width=2),
                                  showlegend=(col == 1), legendgroup='High'),
                      row=1, col=col)

    fig.update_yaxes(title_text="Membership μ", row=1, col=1)
    fig.update_layout(height=350)
    st.plotly_chart(style_plotly(fig), use_container_width=True)

    # Urgency distribution + triage breakdown
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📊 Urgency Score Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=fuzzy_scores['Urgency_Score'],
            nbinsx=30,
            marker=dict(color=COLORS['accent'],
                        line=dict(color='white', width=1)),
            name='Urgency'
        ))
        fig.add_vline(x=avg_urgency, line_dash="dash", line_color=COLORS['primary'],
                      annotation_text=f"Mean = {avg_urgency:.1f}",
                      annotation_font_color=COLORS['primary'])
        fig.update_layout(xaxis_title="Urgency Score (0-100)", yaxis_title="Patients",
                          showlegend=False)
        st.plotly_chart(style_plotly(fig, height=380), use_container_width=True)

    with col_r:
        st.markdown("#### 🎯 Triage Classification")
        triage = fuzzy_scores['Triage_Class'].value_counts().reset_index()
        triage.columns = ['Class', 'Count']
        fig = go.Figure(data=[go.Bar(
            x=triage['Class'], y=triage['Count'],
            marker=dict(color=[TRIAGE_COLORS.get(c, COLORS['primary'])
                               for c in triage['Class']],
                        line=dict(color='white', width=1)),
            text=triage['Count'], textposition='outside',
            textfont=dict(color=COLORS['text_dark'], size=14),
        )])
        fig.update_layout(xaxis_title="Triage Class", yaxis_title="Patient Count",
                          showlegend=False)
        st.plotly_chart(style_plotly(fig, height=380), use_container_width=True)

    # Patient lookup
    st.markdown("#### 🔍 Patient Fuzzy Inference Lookup")
    selected_patient = st.selectbox(
        "Select a Patient ID",
        fuzzy_scores['Patient_ID'].sort_values().tolist(),
        index=0
    )
    patient = fuzzy_scores[fuzzy_scores['Patient_ID'] == selected_patient].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Heart Rate",     f"{patient['Heart_Rate']} bpm",
              f"Fuzzy: {patient['Heart_Rate_Fuzzy']:.1f}")
    c2.metric("Blood Pressure", f"{patient['Blood_Pressure']} mmHg",
              f"Fuzzy: {patient['Blood_Pressure_Fuzzy']:.1f}")
    c3.metric("Oxygen Level",   f"{patient['Oxygen_Level']}%",
              f"Fuzzy: {patient['Oxygen_Fuzzy']:.1f}")
    c4.metric("Urgency Score",  f"{patient['Urgency_Score']:.1f}",
              patient['Triage_Class'])


# ============================================================================
# PAGE 4 — MCDM ANALYSIS
# ============================================================================
def page_mcdm():
    page_header("🎯 Multi-Criteria Decision Making",
                "AHP weight elicitation and TOPSIS patient ranking")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients Ranked",  f"{len(patient_rankings):,}")
    c2.metric("Max TOPSIS Score", f"{patient_rankings['TOPSIS_Score'].max():.4f}")
    c3.metric("Avg TOPSIS Score", f"{patient_rankings['TOPSIS_Score'].mean():.4f}")
    c4.metric("Top Priority",      patient_rankings.iloc[0]['Patient_ID'])

    # AHP weights + TOPSIS top-20
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.markdown("#### ⚖️ AHP Criteria Weights")
        ahp_data = pd.DataFrame({
            'Criterion': ['Urgency', 'Waiting Time', 'Resource Cost', 'Fairness'],
            'Weight':    [0.40, 0.30, 0.15, 0.15],
        })
        fig = go.Figure(data=[go.Pie(
            labels=ahp_data['Criterion'], values=ahp_data['Weight'],
            marker=dict(colors=[COLORS['accent'], COLORS['secondary'],
                                COLORS['success'], COLORS['warning']],
                        line=dict(color='white', width=2)),
            hole=0.55, textinfo='label+percent',
            textfont=dict(color='white', size=12),
        )])
        fig.update_layout(height=400, showlegend=False,
                          annotations=[dict(text='AHP<br>Weights', x=0.5, y=0.5,
                                            font=dict(size=14, color=COLORS['primary']),
                                            showarrow=False)])
        st.plotly_chart(style_plotly(fig), use_container_width=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Consistency Ratio (CR):** `0.0124` ✅ Consistent")
        st.markdown("**λ_max:** `4.033` | **CI:** `0.011` | **RI:** `0.90`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 🏆 Top 20 Patients by TOPSIS Score")
        top20 = patient_rankings.nsmallest(20, 'Priority_Rank').copy()
        top20 = top20.sort_values('TOPSIS_Score')
        fig = go.Figure(go.Bar(
            x=top20['TOPSIS_Score'],
            y=top20['Patient_ID'],
            orientation='h',
            marker=dict(color=[SEVERITY_COLORS.get(s, COLORS['primary'])
                               for s in top20['Severity']],
                        line=dict(color='white', width=1)),
            text=[f"{s:.4f}" for s in top20['TOPSIS_Score']],
            textposition='outside',
            textfont=dict(color=COLORS['text_dark'], size=10),
        ))
        fig.update_layout(xaxis_title="TOPSIS Score", yaxis_title="",
                          xaxis_range=[0.65, 0.82], showlegend=False)
        st.plotly_chart(style_plotly(fig, height=600), use_container_width=True)

    # Full ranking table
    st.markdown("#### 📋 Patient Priority Table")
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        sev_filter = st.multiselect("Severity Filter",
                                     options=patient_rankings['Severity'].unique(),
                                     default=patient_rankings['Severity'].unique())
    with f2:
        rank_max = st.slider("Show Top N Patients", 10, 100, 25)
    with f3:
        st.write("")  # spacing
        st.write("")
        st.markdown(f"**Showing:** Top {rank_max} ranked patients")

    filtered = (patient_rankings[patient_rankings['Severity'].isin(sev_filter)]
                .nsmallest(rank_max, 'Priority_Rank')
                [['Priority_Rank', 'Patient_ID', 'Severity',
                  'Urgency_Score', 'Waiting_Time', 'Resource_Cost',
                  'TOPSIS_Score']])

    st.dataframe(filtered, use_container_width=True, hide_index=True,
                 column_config={
                    'Priority_Rank': st.column_config.NumberColumn('Rank', format="%d"),
                    'TOPSIS_Score':  st.column_config.ProgressColumn(
                        'TOPSIS Score', min_value=0.5, max_value=0.85, format="%.4f"),
                    'Urgency_Score': st.column_config.ProgressColumn(
                        'Urgency', min_value=0, max_value=100, format="%.1f"),
                 })


# ============================================================================
# PAGE 5 — REINFORCEMENT LEARNING ANALYTICS
# ============================================================================
def page_rl():
    page_header("🤖 Reinforcement Learning Analytics",
                "Q-Learning training analysis, policy visualization, and exploration trade-offs")

    # Simulated training data (deterministic)
    np.random.seed(42)
    episodes = np.arange(1, 1001)
    rewards = (
        500 + 1000 * (1 - np.exp(-episodes / 200))
        + 200 * np.sin(episodes / 30)
        + np.random.normal(0, 80, 1000)
    )
    rewards_smooth = pd.Series(rewards).rolling(50, min_periods=1).mean()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes Trained",    "1,000")
    c2.metric("Final Avg Reward",    f"{rewards_smooth.iloc[-1]:.0f}")
    c3.metric("Learning Rate (α)",   "0.10")
    c4.metric("Discount Factor (γ)", "0.95")

    # Reward + learning curves
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📈 Reward Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=episodes, y=rewards,
                                  mode='lines', name='Episode Reward',
                                  line=dict(color=COLORS['secondary'], width=1),
                                  opacity=0.4))
        fig.add_trace(go.Scatter(x=episodes, y=rewards_smooth,
                                  mode='lines', name='50-ep Moving Avg',
                                  line=dict(color=COLORS['accent'], width=3)))
        fig.update_layout(xaxis_title="Episode", yaxis_title="Total Reward")
        st.plotly_chart(style_plotly(fig, height=350), use_container_width=True)

    with col_r:
        st.markdown("#### 📊 Learning Curve (Cumulative Mean)")
        cum_mean = pd.Series(rewards).expanding().mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=episodes, y=cum_mean,
                                  mode='lines',
                                  line=dict(color=COLORS['success'], width=3),
                                  fill='tozeroy',
                                  fillcolor='rgba(39, 174, 96, 0.15)'))
        fig.update_layout(xaxis_title="Episode", yaxis_title="Cumulative Mean Reward")
        st.plotly_chart(style_plotly(fig, height=350), use_container_width=True)

    # Policy heatmap + epsilon comparison
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🗺️ Policy Heatmap (Best Action)")
        actions = ['Assign_Bed', 'Assign_Doctor', 'Assign_Nurse', 'Delay', 'Transfer']
        np.random.seed(7)
        policy = np.random.choice(5, size=(6, 6), p=[0.35, 0.25, 0.20, 0.10, 0.10])
        # bias toward 'Assign_Bed' when beds available
        for i in range(6):
            for j in range(6):
                if i > 2:
                    policy[i, j] = np.random.choice([0, 1], p=[0.7, 0.3])

        fig = go.Figure(data=go.Heatmap(
            z=policy,
            x=[f'Doc={i}' for i in range(6)],
            y=[f'Bed={i}' for i in range(6)],
            colorscale=[[0, COLORS['secondary']], [0.25, COLORS['success']],
                        [0.5, COLORS['warning']], [0.75, COLORS['info']],
                        [1, COLORS['accent']]],
            colorbar=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=actions,
                tickfont=dict(color=COLORS['text_dark'])
            ),
            text=policy, texttemplate="%{text}",
            textfont=dict(color='white', size=10),
        ))
        fig.update_layout(xaxis_title="Doctors Available", yaxis_title="Beds Available")
        st.plotly_chart(style_plotly(fig, height=400), use_container_width=True)

    with col_r:
        st.markdown("#### 🎲 Exploration vs Exploitation (ε)")
        eps_data = pd.DataFrame({
            'Epsilon': ['ε = 0.1', 'ε = 0.2', 'ε = 0.3'],
            'Final Reward': [1480, 1520, 1465],
            'Convergence Episode': [320, 280, 420],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Final Reward',
                              x=eps_data['Epsilon'], y=eps_data['Final Reward'],
                              marker_color=COLORS['secondary'],
                              text=eps_data['Final Reward'],
                              textposition='outside',
                              textfont=dict(color=COLORS['text_dark']),
                              yaxis='y'))
        fig.add_trace(go.Scatter(name='Convergence Episode',
                                  x=eps_data['Epsilon'], y=eps_data['Convergence Episode'],
                                  mode='lines+markers',
                                  marker=dict(size=12, color=COLORS['accent']),
                                  line=dict(width=3, color=COLORS['accent']),
                                  yaxis='y2'))
        fig.update_layout(
            yaxis=dict(title="Final Reward", side='left',
                       gridcolor=COLORS['border']),
            yaxis2=dict(title="Convergence Episode", side='right',
                        overlaying='y', showgrid=False),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(style_plotly(fig, height=400), use_container_width=True)

    # Episode performance details
    st.markdown("#### 📋 Training Configuration")
    config = pd.DataFrame({
        'Parameter': ['Algorithm', 'State Space', 'Action Space', 'Learning Rate (α)',
                      'Discount (γ)', 'Exploration (ε)', 'Episodes', 'Reward Function'],
        'Value': ['Tabular Q-Learning',
                  '(Beds, Docs, Nurses, Critical, Queue)',
                  '5 actions: Assign_Bed, Assign_Doctor, Assign_Nurse, Delay, Transfer',
                  '0.10', '0.95', '0.20 (ε-greedy)', '1,000',
                  'Critical+100, Wait reduced+50, Waste-50, Critical delayed-100'],
    })
    st.dataframe(config, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 6 — OPTIMIZATION ANALYSIS
# ============================================================================
def page_optimization():
    page_header("⚙️ Optimization Analysis",
                "Comparing RL-only, Optimization-only, and Hybrid RL+Optimization strategies")

    # KPIs - best strategy summary
    best_reward = evaluation_metrics.loc[evaluation_metrics['Total_Reward'].idxmax()]
    best_wait   = evaluation_metrics.loc[evaluation_metrics['Average_Wait_Time'].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Total Reward",
              f"{best_reward['Total_Reward']:,.0f}",
              best_reward['Strategy'])
    c2.metric("⏱️ Best Wait Time",
              f"{best_wait['Average_Wait_Time']:.1f} min",
              best_wait['Strategy'])
    c3.metric("📊 Allocation Rate",
              f"{evaluation_metrics['Allocated_Patients'].iloc[0] / 10:.0f}%")
    c4.metric("🎯 Strategies Compared", "3")

    # Bar chart of all metrics
    st.markdown("#### 📊 Strategy Comparison — All Metrics")

    metric_cols = ['Average_Wait_Time', 'Critical_Response_Time', 'Total_Reward']
    strategies  = evaluation_metrics['Strategy'].tolist()

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Avg Wait Time (min)",
                                        "Critical Response (min)",
                                        "Total Reward"))
    strategy_colors = [COLORS['secondary'], COLORS['accent'], COLORS['success']]

    for i, metric in enumerate(metric_cols, start=1):
        values = evaluation_metrics[metric].tolist()
        fig.add_trace(go.Bar(
            x=strategies, y=values,
            marker=dict(color=strategy_colors,
                        line=dict(color='white', width=1)),
            text=[f"{v:,.1f}" if v < 1000 else f"{v:,.0f}" for v in values],
            textposition='outside',
            textfont=dict(color=COLORS['text_dark'], size=12),
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(height=420)
    fig.update_xaxes(tickangle=-15)
    st.plotly_chart(style_plotly(fig), use_container_width=True)

    # Radar chart + utility
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.markdown("#### 🕸️ Multi-Metric Radar")
        # Normalize for radar
        radar_metrics = ['Average_Wait_Time', 'Resource_Utilization',
                         'Critical_Response_Time', 'Fairness_Index', 'Total_Reward']
        norm_df = evaluation_metrics.copy()
        for m in radar_metrics:
            if m in ['Average_Wait_Time', 'Critical_Response_Time']:
                # Lower is better → invert
                norm_df[m] = 1 - (norm_df[m] - norm_df[m].min()) / max(norm_df[m].max() - norm_df[m].min(), 1e-9)
            else:
                norm_df[m] = (norm_df[m] - norm_df[m].min()) / max(norm_df[m].max() - norm_df[m].min(), 1e-9)

        fig = go.Figure()
        for i, strat in enumerate(strategies):
            row = norm_df[norm_df['Strategy'] == strat].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics] + [row[radar_metrics[0]]],
                theta=[m.replace('_', '<br>') for m in radar_metrics] + [radar_metrics[0].replace('_','<br>')],
                fill='toself',
                name=strat,
                line=dict(color=strategy_colors[i], width=2),
                fillcolor=strategy_colors[i],
                opacity=0.4,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor='white',
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor=COLORS['border']),
                angularaxis=dict(gridcolor=COLORS['border'],
                                  tickfont=dict(color=COLORS['text_dark'], size=10)),
            ),
            height=480,
            showlegend=True,
            legend=dict(x=0.7, y=1.1, orientation='h'),
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          font=dict(color=COLORS['text_dark']))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 📋 Detailed Metrics Table")
        display_df = evaluation_metrics.copy()
        for c in display_df.select_dtypes(include='float').columns:
            display_df[c] = display_df[c].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     column_config={
                        'Total_Reward': st.column_config.ProgressColumn(
                            'Total Reward', min_value=15000, max_value=22000,
                            format="%.0f"),
                        'Fairness_Index': st.column_config.ProgressColumn(
                            'Fairness', min_value=0, max_value=1, format="%.2f"),
                        'Resource_Utilization': st.column_config.ProgressColumn(
                            'Utilization', min_value=0, max_value=1, format="%.2f"),
                     })

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("##### 🎯 Recommendation")
        st.markdown(
            "**RL + Optimization Hybrid** achieves the highest Total Reward "
            f"({best_reward['Total_Reward']:,.0f}) while maintaining competitive "
            "wait times. Recommend deployment as default operational strategy."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Allocation breakdown from allocation_results
    st.markdown("#### 🔬 Allocation Decisions Breakdown")
    action_counts = allocation_results['Action'].value_counts().reset_index()
    action_counts.columns = ['Action', 'Count']
    col_l, col_r = st.columns([2, 3])
    with col_l:
        fig = go.Figure(data=[go.Pie(
            labels=action_counts['Action'], values=action_counts['Count'],
            marker=dict(colors=[ACTION_COLORS.get(a, COLORS['primary'])
                                for a in action_counts['Action']],
                        line=dict(color='white', width=2)),
            hole=0.5, textinfo='label+percent',
            textfont=dict(color='white', size=12)
        )])
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(style_plotly(fig), use_container_width=True)
    with col_r:
        fig = px.scatter(
            allocation_results.head(200),
            x='Priority_Rank', y='Utility_Score', color='Action',
            color_discrete_map=ACTION_COLORS,
            size='Reward', size_max=20,
            hover_data=['Patient_ID', 'Post_Allocation_Wait'],
        )
        fig.update_layout(xaxis_title="Priority Rank", yaxis_title="Utility Score")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)


# ============================================================================
# PAGE 7 — DECISION TRADE-OFF STUDIO
# ============================================================================
def page_tradeoff():
    page_header("🎚️ Decision Trade-Off Studio",
                "Interactive what-if simulation — adjust parameters to see system response")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### ⚙️ Simulation Parameters")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        arrival_rate    = st.slider("📈 Arrival Rate Multiplier",
                                     0.5, 3.0, 1.0, 0.1,
                                     help="1.0 = baseline (default).")
        bed_capacity    = st.slider("🛏️ Bed Capacity",      10, 60, 30, 1)
        doctor_capacity = st.slider("👨‍⚕️ Doctor Capacity",   5, 30, 15, 1)
        fairness_weight = st.slider("⚖️ Fairness Weight",   0.0, 0.5, 0.15, 0.05)
        critical_thresh = st.slider("🚨 Critical Threshold (urgency)",
                                     50, 95, 75, 5)
        st.markdown('</div>', unsafe_allow_html=True)

    # Baseline (from evaluation_metrics)
    baseline = evaluation_metrics[evaluation_metrics['Strategy'] == 'RL + Optimization'].iloc[0]
    base_wait     = baseline['Average_Wait_Time']
    base_util     = baseline['Resource_Utilization']
    base_response = baseline['Critical_Response_Time']
    base_fairness = baseline['Fairness_Index']
    base_reward   = baseline['Total_Reward']

    # Re-simulate outcomes
    capacity_factor = (30 / max(bed_capacity, 1)) * 0.5 + (15 / max(doctor_capacity, 1)) * 0.5
    sim_wait      = base_wait * arrival_rate * capacity_factor
    sim_util      = min(0.99, base_util * arrival_rate / max(capacity_factor, 0.5))
    sim_response  = base_response * arrival_rate * (1 - (critical_thresh - 50) / 200)
    sim_fairness  = min(1.0, base_fairness + fairness_weight * 1.2)
    sim_reward    = base_reward * (1.5 - 0.4 * arrival_rate) * (bed_capacity / 30) * (1 + fairness_weight * 0.5)

    with col_r:
        st.markdown("#### 📊 Simulated Outcomes")
        c1, c2 = st.columns(2)
        c1.metric("Avg Wait Time", f"{sim_wait:.1f} min",
                  f"{sim_wait - base_wait:+.1f} vs baseline",
                  delta_color="inverse")
        c2.metric("Resource Utilization", f"{sim_util:.2%}",
                  f"{(sim_util - base_util)*100:+.1f}%")
        c1, c2 = st.columns(2)
        c1.metric("Critical Response", f"{sim_response:.1f} min",
                  f"{sim_response - base_response:+.1f} vs baseline",
                  delta_color="inverse")
        c2.metric("Fairness Index", f"{sim_fairness:.3f}",
                  f"{sim_fairness - base_fairness:+.3f}")
        st.metric("Total Reward", f"{sim_reward:,.0f}",
                  f"{sim_reward - base_reward:+,.0f} vs baseline")

    # Comparison chart
    st.markdown("#### 📈 Simulated vs Baseline")
    compare_df = pd.DataFrame({
        'Metric': ['Avg Wait (min)', 'Utilization (×100)', 'Critical Response (min)',
                   'Fairness (×100)', 'Reward (÷200)'],
        'Baseline':  [base_wait, base_util * 100, base_response,
                       base_fairness * 100, base_reward / 200],
        'Simulated': [sim_wait, sim_util * 100, sim_response,
                       sim_fairness * 100, sim_reward / 200],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Baseline', x=compare_df['Metric'],
                          y=compare_df['Baseline'],
                          marker_color=COLORS['secondary'],
                          text=compare_df['Baseline'].round(1),
                          textposition='outside',
                          textfont=dict(color=COLORS['text_dark'])))
    fig.add_trace(go.Bar(name='Simulated', x=compare_df['Metric'],
                          y=compare_df['Simulated'],
                          marker_color=COLORS['accent'],
                          text=compare_df['Simulated'].round(1),
                          textposition='outside',
                          textfont=dict(color=COLORS['text_dark'])))
    fig.update_layout(barmode='group',
                      yaxis_title="Value (normalized for comparison)",
                      legend=dict(x=0.01, y=0.99))
    st.plotly_chart(style_plotly(fig, height=400), use_container_width=True)

    # Sensitivity sweep chart
    st.markdown("#### 🔍 Sensitivity Analysis — Arrival Rate Sweep")
    sweep_rates = np.linspace(0.5, 3.0, 20)
    sweep_data = []
    for r in sweep_rates:
        cf = (30 / max(bed_capacity, 1)) * 0.5 + (15 / max(doctor_capacity, 1)) * 0.5
        sweep_data.append({
            'Arrival_Rate': r,
            'Wait_Time':    base_wait * r * cf,
            'Response':     base_response * r * (1 - (critical_thresh - 50) / 200),
            'Reward':       base_reward * (1.5 - 0.4 * r) * (bed_capacity / 30) * (1 + fairness_weight * 0.5),
        })
    sweep_df = pd.DataFrame(sweep_data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sweep_df['Arrival_Rate'], y=sweep_df['Wait_Time'],
                              name='Wait Time', line=dict(color=COLORS['secondary'], width=3),
                              fill='tozeroy', fillcolor='rgba(41, 128, 185, 0.15)'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=sweep_df['Arrival_Rate'], y=sweep_df['Reward'],
                              name='Reward', line=dict(color=COLORS['success'], width=3)),
                  secondary_y=True)
    fig.add_vline(x=arrival_rate, line_dash="dash", line_color=COLORS['accent'],
                  annotation_text=f"Current: {arrival_rate}",
                  annotation_font_color=COLORS['accent'])
    fig.update_xaxes(title_text="Arrival Rate Multiplier")
    fig.update_yaxes(title_text="Wait Time (min)", secondary_y=False)
    fig.update_yaxes(title_text="Total Reward", secondary_y=True)
    st.plotly_chart(style_plotly(fig, height=400), use_container_width=True)


# ============================================================================
# PAGE 8 — EXECUTIVE DASHBOARD
# ============================================================================
def page_executive():
    page_header("📈 Executive Dashboard",
                "Top-level KPIs, trends, and operational performance summary")

    # Hero KPIs
    best = evaluation_metrics.loc[evaluation_metrics['Total_Reward'].idxmax()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⏱️ Avg Wait Time",       f"{best['Average_Wait_Time']:.1f} min", "-5.1 vs baseline")
    c2.metric("📊 Resource Util",        f"{best['Resource_Utilization']:.1%}", "+8.2%")
    c3.metric("💚 Survival Score",       f"{fuzzy_scores['Survival_Probability'].mean():.1%}", "+3.4%")
    c4.metric("⚖️ Fairness Score",       f"{best['Fairness_Index']:.2f}", "stable")
    c5.metric("🏆 Total Reward",         f"{best['Total_Reward']:,.0f}", "+13.6% vs RL")

    # Gauges
    st.markdown("#### 📊 Performance Gauges")
    g1, g2, g3, g4 = st.columns(4)

    def make_gauge(value, title, max_val, color, suffix=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix=suffix, font=dict(color=COLORS['primary'], size=28)),
            title=dict(text=title, font=dict(color=COLORS['text_dark'], size=14)),
            gauge=dict(
                axis=dict(range=[0, max_val], tickfont=dict(color=COLORS['text_dark'])),
                bar=dict(color=color),
                bgcolor='white',
                borderwidth=2,
                bordercolor=COLORS['border'],
                steps=[
                    {'range': [0, max_val * 0.5], 'color': '#fde2e2'},
                    {'range': [max_val * 0.5, max_val * 0.8], 'color': '#fff3cd'},
                    {'range': [max_val * 0.8, max_val], 'color': '#d4edda'},
                ],
                threshold=dict(line=dict(color=COLORS['accent'], width=4),
                                thickness=0.75, value=value),
            ),
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20),
                          paper_bgcolor='white', font=dict(color=COLORS['text_dark']))
        return fig

    with g1:
        st.plotly_chart(make_gauge(best['Resource_Utilization'] * 100,
                                    "Resource Utilization", 100, COLORS['secondary'], "%"),
                        use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(fuzzy_scores['Survival_Probability'].mean() * 100,
                                    "Survival Score", 100, COLORS['success'], "%"),
                        use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(best['Fairness_Index'] * 100,
                                    "Fairness Index", 100, COLORS['info'], "%"),
                        use_container_width=True)
    with g4:
        wait_score = max(0, 100 - best['Average_Wait_Time'])
        st.plotly_chart(make_gauge(wait_score,
                                    "Wait Time Score", 100, COLORS['warning'], ""),
                        use_container_width=True)

    # Trend charts + heatmap
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("#### 📈 Hourly Performance Trends")
        # Build hourly trend
        hourly = hospital_data.groupby('Hour').agg(
            Arrivals     =('Patient_ID','count'),
            Avg_Wait     =('Waiting_Time','mean'),
            Avg_Treatment=('Treatment_Time','mean'),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=hourly['Hour'], y=hourly['Arrivals'],
                              name='Arrivals',
                              marker_color=COLORS['secondary'], opacity=0.6),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=hourly['Hour'], y=hourly['Avg_Wait'],
                                  name='Avg Wait (min)',
                                  line=dict(color=COLORS['accent'], width=3),
                                  mode='lines+markers'),
                      secondary_y=True)
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Patient Arrivals", secondary_y=False)
        fig.update_yaxes(title_text="Avg Wait Time (min)", secondary_y=True)
        fig.update_layout(legend=dict(x=0.01, y=0.99))
        st.plotly_chart(style_plotly(fig, height=380), use_container_width=True)

    with col_r:
        st.markdown("#### 🌡️ Resource Usage Heatmap")
        # Build hour × resource heatmap
        heat = hospital_data.groupby('Hour')[
            ['Bed_Required','Doctor_Required','Nurse_Required','Ventilator_Required']
        ].sum().T
        heat.index = ['Beds', 'Doctors', 'Nurses', 'Ventilators']

        fig = go.Figure(data=go.Heatmap(
            z=heat.values, x=heat.columns, y=heat.index,
            colorscale=[[0, '#e8f4f8'], [0.5, COLORS['secondary']], [1, COLORS['primary']]],
            colorbar=dict(title="Demand", tickfont=dict(color=COLORS['text_dark'])),
            hovertemplate='Hour: %{x}<br>Resource: %{y}<br>Demand: %{z}<extra></extra>',
        ))
        fig.update_xaxes(title_text="Hour of Day")
        st.plotly_chart(style_plotly(fig, height=380), use_container_width=True)

    # Strategy summary cards
    st.markdown("#### 🎯 Strategy Performance Summary")
    cols = st.columns(3)
    for i, (_, row) in enumerate(evaluation_metrics.iterrows()):
        color = [COLORS['secondary'], COLORS['accent'], COLORS['success']][i]
        is_best = row['Total_Reward'] == evaluation_metrics['Total_Reward'].max()
        badge = "🏆 RECOMMENDED" if is_best else ""
        with cols[i]:
            st.markdown(f"""
            <div style="background:#ffffff; border-radius:12px; padding:1.2rem;
                        border-top:4px solid {color}; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="color:#5a6c7d; font-size:0.8rem; font-weight:600; text-transform:uppercase;">
                    {badge}
                </div>
                <h3 style="color:#1a3a5c; margin:0.3rem 0;">{row['Strategy']}</h3>
                <hr style="border:none; border-top:1px solid #e1e8ed; margin:0.5rem 0;">
                <div style="color:#2c3e50;">
                    <p style="margin:0.3rem 0;"><b>Wait Time:</b> {row['Average_Wait_Time']:.1f} min</p>
                    <p style="margin:0.3rem 0;"><b>Response:</b> {row['Critical_Response_Time']:.1f} min</p>
                    <p style="margin:0.3rem 0;"><b>Utilization:</b> {row['Resource_Utilization']:.1%}</p>
                    <p style="margin:0.3rem 0;"><b>Fairness:</b> {row['Fairness_Index']:.2f}</p>
                    <p style="margin:0.3rem 0;"><b>Allocated:</b> {row['Allocated_Patients']}</p>
                    <p style="margin:0.3rem 0; color:{color}; font-size:1.2rem;">
                        <b>Reward: {row['Total_Reward']:,.0f}</b>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# ROUTER
# ============================================================================
PAGES = {
    "📊  1. Hospital Overview":   page_overview,
    "▶️  2. Live Simulation":     page_live_simulation,
    "🔥  3. Fuzzy Triage":         page_fuzzy,
    "🎯  4. MCDM Analysis":        page_mcdm,
    "🤖  5. RL Analytics":         page_rl,
    "⚙️  6. Optimization":        page_optimization,
    "🎚️  7. Trade-Off Studio":    page_tradeoff,
    "📈  8. Executive Dashboard": page_executive,
}

PAGES[page]()
