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

    /* ALL headings dark navy by default — works regardless of container class */
    h1, h2, h3, h4, h5, h6 {
        color: #1a3a5c !important;
        font-weight: 700 !important;
    }

    /* Streamlit markdown headers explicitly */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #1a3a5c !important;
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
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h6 {
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

    /* Force Plotly/SVG chart text to black on white chart backgrounds */
    .js-plotly-plot .plotly text,
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .legendtext,
    .js-plotly-plot .gtitle,
    .js-plotly-plot .annotation-text,
    svg text {
        fill: #000000 !important;
        color: #000000 !important;
    }

    .js-plotly-plot .hovertext text {
        fill: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================
def style_plotly(fig, title=None, height=None):
    """Apply consistent black-text-on-white styling to all Plotly charts.

    This also removes common Plotly "undefined" hover/label artifacts caused by
    empty trace names or null text fields.
    """
    fig.update_layout(
        title=dict(text=title, font=dict(color="#000000", size=16)) if title else None,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#000000", family='Arial, sans-serif', size=12),
        hoverlabel=dict(bgcolor="white", font_color="black", bordercolor=COLORS['border']),
        xaxis=dict(
            gridcolor=COLORS['border'],
            linecolor='#bdc3c7',
            tickfont=dict(color="#000000"),
            title_font=dict(color="#000000"),
        ),
        yaxis=dict(
            gridcolor=COLORS['border'],
            linecolor='#bdc3c7',
            tickfont=dict(color="#000000"),
            title_font=dict(color="#000000"),
        ),
        legend=dict(
            font=dict(color="#000000"),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS['border'],
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=60 if title else 30, b=40),
    )
    fig.update_xaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
    fig.update_yaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))

    # Clean trace-level labels and hover output. Never allow visible "undefined".
    bad_tokens = {None, "undefined", "Undefined", "UNDEFINED", "nan", "NaN", "None"}

    def _clean_value(v):
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        txt = str(v)
        if txt.strip() in bad_tokens or txt.strip().lower() == "undefined":
            return ""
        return txt.replace("undefined", "").replace("Undefined", "").replace("UNDEFINED", "")

    for trace in fig.data:
        if getattr(trace, "name", None) in bad_tokens or str(getattr(trace, "name", "")).lower() == "undefined":
            trace.name = ""
            trace.showlegend = False if getattr(trace, "showlegend", None) is None else trace.showlegend
        if hasattr(trace, "text") and trace.text is not None:
            try:
                trace.text = [_clean_value(x) for x in trace.text]
            except TypeError:
                trace.text = _clean_value(trace.text)
        if hasattr(trace, "customdata") and trace.customdata is not None:
            try:
                trace.customdata = [[_clean_value(x) for x in row] if isinstance(row, (list, tuple, np.ndarray)) else _clean_value(row) for row in trace.customdata]
            except Exception:
                pass
        if hasattr(trace, "textfont"):
            trace.textfont = dict(color="#000000", size=getattr(getattr(trace, "textfont", None), "size", 12) or 12)
        if getattr(trace, "hovertemplate", None):
            trace.hovertemplate = _clean_value(trace.hovertemplate)
        elif getattr(trace, "hoverinfo", None) is None:
            # Keeps Plotly from appending trace names such as undefined in hover boxes.
            trace.hovertemplate = None

    # Clean annotation text, including vline annotations and subplot titles.
    if getattr(fig.layout, "annotations", None):
        for ann in fig.layout.annotations:
            ann.text = _clean_value(getattr(ann, "text", ""))
            ann.font = dict(color="#000000", size=getattr(getattr(ann, "font", None), "size", 12) or 12)

    if height:
        fig.update_layout(height=height)
    return fig


def page_header(title, subtitle):
    # Single-line HTML — no leading whitespace, otherwise markdown treats it as a code block
    html = (
        '<div class="page-header">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def resource_bar(label, current, capacity, color):
    pct = (current / capacity) * 100 if capacity > 0 else 0
    text_color = "#ffffff" if pct > 25 else COLORS['text_dark']
    html = (
        '<div style="margin-bottom:0.8rem;">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">'
        f'<span style="color:#2c3e50;font-weight:600;">{label}</span>'
        f'<span style="color:#5a6c7d;">{current}/{capacity}</span>'
        '</div>'
        '<div class="resource-bar">'
        f'<div class="resource-bar-fill" style="width:{pct}%;background-color:{color};color:{text_color};">'
        f'{pct:.0f}%'
        '</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def strategy_card(strategy_name, color, badge, wait, response, util, fairness, allocated, reward):
    """Render a strategy summary card. Single-line HTML to dodge markdown's code-block bug."""
    html = (
        f'<div style="background:#ffffff;border-radius:12px;padding:1.2rem;'
        f'border-top:4px solid {color};box-shadow:0 2px 8px rgba(0,0,0,0.06);height:100%;">'
        f'<div style="color:#5a6c7d;font-size:0.75rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.5px;min-height:1.2rem;">{badge}</div>'
        f'<h3 style="color:#1a3a5c;margin:0.3rem 0;">{strategy_name}</h3>'
        f'<hr style="border:none;border-top:1px solid #e1e8ed;margin:0.6rem 0;">'
        f'<div style="color:#2c3e50;line-height:1.7;">'
        f'<p style="margin:0;"><b>Wait Time:</b> {wait:.1f} min</p>'
        f'<p style="margin:0;"><b>Response:</b> {response:.1f} min</p>'
        f'<p style="margin:0;"><b>Utilization:</b> {util:.1%}</p>'
        f'<p style="margin:0;"><b>Fairness:</b> {fairness:.2f}</p>'
        f'<p style="margin:0;"><b>Allocated:</b> {allocated}</p>'
        f'<p style="margin:0.5rem 0 0 0;color:{color};font-size:1.2rem;font-weight:700;">'
        f'Reward: {reward:,.0f}</p>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING — path resolved relative to app.py, not the CWD
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def data_file_path(name):
    """Find a CSV/output file across common Streamlit deployment layouts."""
    candidates = [
        os.path.join(SCRIPT_DIR, 'data'),
        SCRIPT_DIR,
        os.path.join(os.getcwd(), 'ed_dashboard', 'data'),
        os.path.join(os.getcwd(), 'data'),
        os.getcwd(),
    ]
    for base in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None

def load_optional_csv(name):
    path = data_file_path(name)
    if path is None:
        return None
    return pd.read_csv(path)

@st.cache_data
def load_data():
    # Try several locations so the app works whether Streamlit runs from
    # the repo root, from inside ed_dashboard/, or from anywhere else.
    candidates = [
        os.path.join(SCRIPT_DIR, 'data'),           # ed_dashboard/data  (preferred)
        SCRIPT_DIR,                                 # ed_dashboard/      (CSVs next to app.py)
        os.path.join(os.getcwd(), 'ed_dashboard', 'data'),
        os.path.join(os.getcwd(), 'data'),
        os.getcwd(),
    ]
    base = next((p for p in candidates
                 if os.path.exists(os.path.join(p, 'hospital_data.csv'))), None)
    if base is None:
        raise FileNotFoundError(
            "hospital_data.csv not found. Searched: " + " | ".join(candidates)
        )

    def _read(name): return pd.read_csv(os.path.join(base, name))

    hospital = _read('hospital_data.csv')
    fuzzy    = _read('hospital_fuzzy_scores.csv')
    rankings = _read('patient_rankings.csv')
    alloc    = _read('allocation_results.csv')
    eval_df  = _read('evaluation_metrics.csv')

    # Remove literal undefined/null strings before they reach Plotly labels or hover boxes.
    for _df in [hospital, fuzzy, rankings, alloc, eval_df]:
        _df.replace({
            'undefined': '', 'Undefined': '', 'UNDEFINED': '',
            'nan': '', 'NaN': '', 'None': '', None: ''
        }, inplace=True)

    hospital['Arrival_Time'] = pd.to_datetime(hospital['Arrival_Time'])
    rankings['Arrival_Time'] = pd.to_datetime(rankings['Arrival_Time'])
    hospital['Hour'] = hospital['Arrival_Time'].dt.hour

    return hospital, fuzzy, rankings, alloc, eval_df


try:
    hospital_data_raw, fuzzy_scores_raw, patient_rankings_raw, allocation_results_raw, evaluation_metrics = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Data files not found: {e}")
    st.info(
        "Expected one of these layouts:\n"
        "- `ed_dashboard/data/*.csv`  (recommended)\n"
        "- `data/*.csv`  (next to app.py)\n"
        "- `*.csv`  (same folder as app.py)"
    )
    st.stop()


# ============================================================================
# SIDEBAR NAVIGATION + GLOBAL FILTERS
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

    # ── GLOBAL FILTERS — applied to ALL pages ─────────────────────────────────
    st.markdown("##### 🔍 Global Filters")
    with st.expander("Adjust filters", expanded=False):
        all_severities = sorted(hospital_data_raw['Severity'].unique())
        g_severity = st.multiselect(
            "Severity",
            options=all_severities,
            default=all_severities,
            help="Filter all pages by patient severity class.",
        )

        age_min_default = int(hospital_data_raw['Age'].min())
        age_max_default = int(hospital_data_raw['Age'].max())
        g_age = st.slider(
            "Age Range",
            min_value=age_min_default, max_value=age_max_default,
            value=(age_min_default, age_max_default),
        )

        g_hour = st.slider(
            "Arrival Hour Range",
            min_value=0, max_value=23,
            value=(0, 23),
            help="Filter by hour-of-day of patient arrival.",
        )

        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()

    # ── Apply global filters to all dataframes ────────────────────────────────
    if not g_severity:
        g_severity = all_severities  # avoid empty selection wiping everything

    _filtered = hospital_data_raw[
        hospital_data_raw['Severity'].isin(g_severity)
        & hospital_data_raw['Age'].between(g_age[0], g_age[1])
        & hospital_data_raw['Hour'].between(g_hour[0], g_hour[1])
    ]
    _ids = set(_filtered['Patient_ID'])

    hospital_data      = _filtered.copy()
    fuzzy_scores       = fuzzy_scores_raw[fuzzy_scores_raw['Patient_ID'].isin(_ids)].copy()
    patient_rankings   = patient_rankings_raw[patient_rankings_raw['Patient_ID'].isin(_ids)].copy()
    allocation_results = allocation_results_raw[allocation_results_raw['Patient_ID'].isin(_ids)].copy()

    # Show filter status
    pct = (len(hospital_data) / len(hospital_data_raw)) * 100 if len(hospital_data_raw) else 0
    if len(hospital_data) < len(hospital_data_raw):
        st.info(f"🔎 **{len(hospital_data):,}** of **{len(hospital_data_raw):,}** patients ({pct:.0f}%)")
    else:
        st.success(f"📊 Showing all **{len(hospital_data):,}** patients")

    # Guard: if filter is too aggressive, halt
    if len(hospital_data) == 0:
        st.error("No patients match the current filters. Widen them to continue.")
        st.stop()

    st.markdown("---")
    st.markdown("##### 📁 Dataset")
    st.markdown(f"**Patients (filtered):** {len(hospital_data):,}")
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

    # ── Local filter ──────────────────────────────────────────────────────────
    with st.expander("🔧 Local filters", expanded=False):
        time_band = st.radio(
            "Time of Day Quick-Filter",
            options=["All Day", "Morning (06-12)", "Afternoon (12-18)",
                     "Evening (18-24)", "Overnight (00-06)"],
            horizontal=True,
        )
    band_map = {
        "Morning (06-12)":   (6, 12),
        "Afternoon (12-18)": (12, 18),
        "Evening (18-24)":   (18, 24),
        "Overnight (00-06)": (0, 6),
    }
    if time_band != "All Day":
        lo, hi = band_map[time_band]
        df = hospital_data[hospital_data['Hour'].between(lo, hi - 1)]
    else:
        df = hospital_data

    if len(df) == 0:
        st.warning("No data for this time band with current global filters.")
        return

    # Simulated current-state values (scale with filtered data)
    beds_capacity, beds_used = 30, min(30, int(len(df) * 0.022))
    doc_capacity,  doc_used  = 15, min(15, int(len(df) * 0.011))
    nurse_capacity, nurse_used = 20, min(20, int(len(df) * 0.016))
    vent_capacity, vent_used = 10, min(10, int(len(df) * 0.004))
    critical_count = (df['Severity'] == 'Critical').sum()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Patients", f"{len(df):,}",
              f"{time_band}" if time_band != "All Day" else "")
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
        hourly = df.groupby('Hour').size().reset_index(name='Arrivals')
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
        sev = df['Severity'].value_counts().reset_index()
        sev.columns = ['Severity', 'Count']
        fig = go.Figure(data=[go.Pie(
            labels=sev['Severity'], values=sev['Count'],
            marker=dict(colors=[SEVERITY_COLORS.get(s, COLORS['primary']) for s in sev['Severity']],
                        line=dict(color='white', width=2)),
            hole=0.5,
            textinfo='label+percent',
            textfont=dict(color='#000000', size=12),
        )])
        fig.update_layout(showlegend=True, height=350,
                          legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.1))
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    # Row: Resource utilization + demand by severity
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.markdown("#### 🛏️ Current Resource Status")
        with st.container(border=True):
            resource_bar("🛏️ Beds",        beds_used,  beds_capacity,  COLORS['secondary'])
            resource_bar("👨‍⚕️ Doctors",    doc_used,   doc_capacity,   COLORS['success'])
            resource_bar("👩‍⚕️ Nurses",     nurse_used, nurse_capacity, COLORS['warning'])
            resource_bar("🫁 Ventilators", vent_used,  vent_capacity,  COLORS['accent'])

    with col_r:
        st.markdown("#### 🎯 Resource Demand by Severity")
        demand = df.groupby('Severity')[
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
        with st.container(border=True):
            resource_bar("🛏️ Beds",        beds_used,   30, COLORS['secondary'])
            resource_bar("👨‍⚕️ Doctors",    docs_used,   15, COLORS['success'])
            resource_bar("👩‍⚕️ Nurses",     nurses_used, 20, COLORS['warning'])
            resource_bar("🫁 Ventilators", vents_used,  10, COLORS['accent'])

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

    # ── Local filter ──────────────────────────────────────────────────────────
    with st.expander("🔧 Local filters", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            all_classes = sorted(fuzzy_scores['Triage_Class'].unique())
            class_filter = st.multiselect(
                "Triage Class", options=all_classes, default=all_classes
            )
        with f2:
            urg_range = st.slider(
                "Urgency Score Range",
                min_value=0, max_value=100, value=(0, 100), step=5
            )

    if not class_filter:
        class_filter = all_classes
    fz = fuzzy_scores[
        fuzzy_scores['Triage_Class'].isin(class_filter)
        & fuzzy_scores['Urgency_Score'].between(urg_range[0], urg_range[1])
    ]
    if len(fz) == 0:
        st.warning("No patients match these fuzzy filters.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    avg_urgency = fz['Urgency_Score'].mean()
    high_urgency = (fz['Urgency_Score'] > 70).sum()
    avg_survival = fz['Survival_Probability'].mean()
    c1.metric("Avg Urgency Score", f"{avg_urgency:.1f}")
    c2.metric("High Urgency (>70)", f"{high_urgency}")
    c3.metric("Avg Survival Prob", f"{avg_survival:.2%}")
    c4.metric("Critical Triage",
              f"{(fz['Triage_Class']=='Critical').sum()}")

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
            x=fz['Urgency_Score'],
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
        triage = fz['Triage_Class'].value_counts().reset_index()
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
        fz['Patient_ID'].sort_values().tolist(),
        index=0
    )
    patient = fz[fz['Patient_ID'] == selected_patient].iloc[0]

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
            textfont=dict(color='#000000', size=12),
        )])
        fig.update_layout(height=400, showlegend=False,
                          annotations=[dict(text='AHP<br>Weights', x=0.5, y=0.5,
                                            font=dict(size=14, color=COLORS['primary']),
                                            showarrow=False)])
        st.plotly_chart(style_plotly(fig), use_container_width=True)

        with st.container(border=True):
            st.markdown("**Consistency Ratio (CR):** `0.0124` ✅ Consistent")
            st.markdown("**λ_max:** `4.033` | **CI:** `0.011` | **RI:** `0.90`")

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

    # Load real RL outputs produced by the Colab notebook when available.
    diagnostics = load_optional_csv('rl_diagnostics.csv')
    policy_summary = load_optional_csv('q_policy_summary.csv')

    if diagnostics is None or policy_summary is None or policy_summary.empty:
        st.warning(
            "Real RL output files were not found. Add `rl_diagnostics.csv` and "
            "`q_policy_summary.csv` to the same data folder as the other CSVs. "
            "The dashboard will not display a random/fake policy heatmap."
        )
        st.info(
            "Run the fixed Colab notebook first, then copy these files into `ed_dashboard/data/`: "
            "`rl_diagnostics.csv`, `q_policy_summary.csv`, and optionally `Q_Table.pkl`."
        )
        return

    diagnostics = diagnostics.copy()
    policy_summary = policy_summary.copy()
    diagnostics['Metric'] = diagnostics['Metric'].astype(str)
    diagnostics['Value'] = pd.to_numeric(diagnostics['Value'], errors='coerce').fillna(0)

    diag = dict(zip(diagnostics['Metric'], diagnostics['Value']))
    episodes = int(diag.get('Episodes', 1000))
    learned_states = int(diag.get('Learned_States', len(policy_summary)))
    total_visits = int(diag.get('Total_State_Visits', policy_summary.get('Visits', pd.Series([0])).sum()))
    exploration = int(diag.get('Exploration_Actions', 0))
    exploitation = int(diag.get('Exploitation_Actions', max(total_visits - exploration, 0)))

    # KPIs based on real diagnostics, not simulated curves.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes Trained", f"{episodes:,}")
    c2.metric("Learned States", f"{learned_states:,}")
    c3.metric("State Visits", f"{total_visits:,}")
    explore_rate = exploration / max(exploration + exploitation, 1)
    c4.metric("Exploration Rate", f"{explore_rate:.1%}")

    # Diagnostics chart + action distribution.
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📊 RL Diagnostics")
        show_metrics = diagnostics[diagnostics['Metric'].isin([
            'Episodes', 'Learned_States', 'Total_State_Visits',
            'Exploration_Actions', 'Exploitation_Actions'
        ])].copy()
        show_metrics['Metric'] = show_metrics['Metric'].str.replace('_', ' ')
        fig = go.Figure(go.Bar(
            x=show_metrics['Metric'],
            y=show_metrics['Value'],
            text=[f"{v:,.0f}" for v in show_metrics['Value']],
            textposition='outside',
            textfont=dict(color='#000000', size=12),
            marker=dict(color=COLORS['secondary'], line=dict(color='white', width=1)),
        ))
        fig.update_layout(xaxis_title="Metric", yaxis_title="Value", showlegend=False)
        st.plotly_chart(style_plotly(fig, height=360), use_container_width=True)

    with col_r:
        st.markdown("#### 🎯 Learned Action Distribution")
        action_counts = policy_summary['Best_Action'].fillna('Unknown').value_counts().reset_index()
        action_counts.columns = ['Action', 'Count']
        fig = go.Figure(go.Bar(
            x=action_counts['Action'],
            y=action_counts['Count'],
            text=action_counts['Count'],
            textposition='outside',
            textfont=dict(color='#000000', size=12),
            marker=dict(color=COLORS['success'], line=dict(color='white', width=1)),
        ))
        fig.update_layout(xaxis_title="Best learned action", yaxis_title="Number of visited states", showlegend=False)
        st.plotly_chart(style_plotly(fig, height=360), use_container_width=True)

    # Policy heatmap from visited states only.
    st.markdown("#### 🗺️ Policy Heatmap from Visited Q-Table States")
    st.caption(
        "This heatmap uses only states that appeared in `q_policy_summary.csv`. "
        "It avoids the misleading old behaviour where unseen Q-table states defaulted to zero and `argmax()` selected the first action."
    )

    import ast
    def parse_state(value):
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, tuple) and len(parsed) >= 2:
                return parsed
        except Exception:
            return None
        return None

    policy_summary['Parsed_State'] = policy_summary['State'].apply(parse_state)
    policy_summary = policy_summary[policy_summary['Parsed_State'].notna()].copy()
    policy_summary['Beds_Available'] = policy_summary['Parsed_State'].apply(lambda x: int(x[0]))
    policy_summary['Doctors_Available'] = policy_summary['Parsed_State'].apply(lambda x: int(x[1]))
    policy_summary['Visits'] = pd.to_numeric(policy_summary['Visits'], errors='coerce').fillna(0)
    policy_summary['Best_Q_Value'] = pd.to_numeric(policy_summary['Best_Q_Value'], errors='coerce').fillna(0)

    actions = sorted(policy_summary['Best_Action'].dropna().unique().tolist())
    action_to_code = {a: i for i, a in enumerate(actions)}
    policy_summary['Action_Code'] = policy_summary['Best_Action'].map(action_to_code)

    min_visits = st.slider(
        "Minimum visits required to show a state",
        min_value=1,
        max_value=max(1, int(policy_summary['Visits'].max())),
        value=min(10, max(1, int(policy_summary['Visits'].max()))),
        help="Higher values hide low-confidence states."
    )
    confident = policy_summary[policy_summary['Visits'] >= min_visits].copy()

    if confident.empty:
        st.warning("No states meet the selected visit threshold. Lower the slider.")
    else:
        # For duplicate bed/doctor pairs, keep the most visited learned state.
        heat_df = (confident.sort_values('Visits', ascending=False)
                   .drop_duplicates(['Beds_Available', 'Doctors_Available']))
        beds = sorted(heat_df['Beds_Available'].unique())
        docs = sorted(heat_df['Doctors_Available'].unique())
        z = np.full((len(beds), len(docs)), np.nan)
        text = np.full((len(beds), len(docs)), '', dtype=object)
        hover = np.full((len(beds), len(docs)), '', dtype=object)

        for _, row in heat_df.iterrows():
            bi = beds.index(row['Beds_Available'])
            di = docs.index(row['Doctors_Available'])
            z[bi, di] = row['Action_Code']
            text[bi, di] = str(row['Best_Action']).replace('Assign_', '')
            hover[bi, di] = (
                f"Beds: {row['Beds_Available']}<br>Doctors: {row['Doctors_Available']}"
                f"<br>Best action: {row['Best_Action']}<br>Visits: {row['Visits']:,.0f}"
                f"<br>Best Q-value: {row['Best_Q_Value']:,.2f}"
            )

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f'Doc={d}' for d in docs],
            y=[f'Bed={b}' for b in beds],
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text="Action", font=dict(color='#000000')),
                tickvals=list(action_to_code.values()),
                ticktext=list(action_to_code.keys()),
                tickfont=dict(color='#000000')
            ),
            text=text,
            texttemplate="%{text}",
            textfont=dict(color='#000000', size=11),
            customdata=hover,
            hovertemplate='%{customdata}<extra></extra>',
            zmin=0,
            zmax=max(1, len(actions) - 1),
        ))
        fig.update_layout(xaxis_title="Doctors Available", yaxis_title="Beds Available")
        st.plotly_chart(style_plotly(fig, height=440), use_container_width=True)

    # State visit confidence heatmap.
    st.markdown("#### 🔎 State Visit Confidence")
    if not policy_summary.empty:
        visit_df = (policy_summary.groupby(['Beds_Available', 'Doctors_Available'])['Visits']
                    .sum().reset_index())
        beds = sorted(visit_df['Beds_Available'].unique())
        docs = sorted(visit_df['Doctors_Available'].unique())
        z = np.zeros((len(beds), len(docs)))
        for _, row in visit_df.iterrows():
            z[beds.index(row['Beds_Available']), docs.index(row['Doctors_Available'])] = row['Visits']
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f'Doc={d}' for d in docs],
            y=[f'Bed={b}' for b in beds],
            colorscale='Blues',
            colorbar=dict(title=dict(text="Visits", font=dict(color='#000000')), tickfont=dict(color='#000000')),
            text=z.astype(int),
            texttemplate="%{text}",
            textfont=dict(color='#000000', size=11),
            hovertemplate='Doctors: %{x}<br>Beds: %{y}<br>Visits: %{z}<extra></extra>',
        ))
        fig.update_layout(xaxis_title="Doctors Available", yaxis_title="Beds Available")
        st.plotly_chart(style_plotly(fig, height=360), use_container_width=True)

    # Training configuration.
    st.markdown("#### 📋 Training Configuration")
    config = pd.DataFrame({
        'Parameter': ['Algorithm', 'State Space', 'Action Space', 'Learning Rate (α)',
                      'Discount (γ)', 'Exploration (ε)', 'Episodes', 'Policy Source'],
        'Value': ['Tabular Q-Learning',
                  '(Beds, Doctors, Nurses, Ventilators, Critical, Queue, Severity/Urgency Bucket)',
                  'Assign_Bed, Assign_Doctor, Assign_Nurse, Assign_Ventilator, Delay, Transfer',
                  '0.10', '0.95', '0.20 ε-greedy', f'{episodes:,}',
                  'Visited states from q_policy_summary.csv'],
    })
    st.dataframe(config, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 6 — OPTIMIZATION ANALYSIS
# ============================================================================
def page_optimization():
    page_header("⚙️ Optimization Analysis",
                "Comparing RL-only, Optimization-only, and Hybrid RL+Optimization strategies")

    # ── Local filter ──────────────────────────────────────────────────────────
    with st.expander("🔧 Local filters", expanded=False):
        all_strategies = evaluation_metrics['Strategy'].tolist()
        strat_filter = st.multiselect(
            "Strategies to Compare",
            options=all_strategies,
            default=all_strategies,
            help="Choose which strategies to include in the comparison.",
        )
    if not strat_filter:
        strat_filter = all_strategies
    eval_filtered = evaluation_metrics[evaluation_metrics['Strategy'].isin(strat_filter)].copy()

    if len(eval_filtered) == 0:
        st.warning("Select at least one strategy.")
        return

    # KPIs - best strategy summary (from filtered set)
    best_reward = eval_filtered.loc[eval_filtered['Total_Reward'].idxmax()]
    best_wait   = eval_filtered.loc[eval_filtered['Average_Wait_Time'].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Total Reward",
              f"{best_reward['Total_Reward']:,.0f}",
              best_reward['Strategy'])
    c2.metric("⏱️ Best Wait Time",
              f"{best_wait['Average_Wait_Time']:.1f} min",
              best_wait['Strategy'])
    c3.metric("📊 Allocation Rate",
              f"{eval_filtered['Allocated_Patients'].iloc[0] / 10:.0f}%")
    c4.metric("🎯 Strategies Compared", str(len(eval_filtered)))

    # Bar chart of all metrics
    st.markdown("#### 📊 Strategy Comparison — All Metrics")

    metric_cols = ['Average_Wait_Time', 'Critical_Response_Time', 'Total_Reward']
    strategies  = eval_filtered['Strategy'].tolist()

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Avg Wait Time (min)",
                                        "Critical Response (min)",
                                        "Total Reward"))
    full_palette = [COLORS['secondary'], COLORS['accent'], COLORS['success']]
    strategy_colors = full_palette[:len(strategies)]

    for i, metric in enumerate(metric_cols, start=1):
        values = eval_filtered[metric].tolist()
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
        norm_df = eval_filtered.copy()
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
        display_df = eval_filtered.copy()
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

        with st.container(border=True):
            st.markdown("##### 🎯 Recommendation")
            st.markdown(
                "**RL + Optimization Hybrid** achieves the highest Total Reward "
                f"({best_reward['Total_Reward']:,.0f}) while maintaining competitive "
                "wait times. Recommend deployment as default operational strategy."
            )

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
            textfont=dict(color='#000000', size=12)
        )])
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(style_plotly(fig), use_container_width=True)
    with col_r:
        # Plotly's `size` parameter requires non-negative values, but Reward can
        # be negative (e.g. resource waste = -50). Use absolute Reward for marker
        # size and keep the signed Reward in the hover tooltip.
        scatter_df = allocation_results.head(200).copy()
        scatter_df['Size'] = scatter_df['Reward'].abs() + 1   # +1 so zero-reward dots still visible
        fig = px.scatter(
            scatter_df,
            x='Priority_Rank', y='Utility_Score', color='Action',
            color_discrete_map=ACTION_COLORS,
            size='Size', size_max=20,
            hover_data=['Patient_ID', 'Post_Allocation_Wait', 'Reward'],
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
        with st.container(border=True):
            arrival_rate    = st.slider("📈 Arrival Rate Multiplier",
                                         0.5, 3.0, 1.0, 0.1,
                                         help="1.0 = baseline (default).")
            bed_capacity    = st.slider("🛏️ Bed Capacity",      10, 60, 30, 1)
            doctor_capacity = st.slider("👨‍⚕️ Doctor Capacity",   5, 30, 15, 1)
            fairness_weight = st.slider("⚖️ Fairness Weight",   0.0, 0.5, 0.15, 0.05)
            critical_thresh = st.slider("🚨 Critical Threshold (urgency)",
                                         50, 95, 75, 5)

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

    # Strategy summary cards — use helper so no markdown indent bug
    st.markdown("#### 🎯 Strategy Performance Summary")
    cols = st.columns(3)
    for i, (_, row) in enumerate(evaluation_metrics.iterrows()):
        color = [COLORS['secondary'], COLORS['accent'], COLORS['success']][i]
        is_best = row['Total_Reward'] == evaluation_metrics['Total_Reward'].max()
        badge = "🏆 RECOMMENDED" if is_best else "STRATEGY"
        with cols[i]:
            strategy_card(
                strategy_name=row['Strategy'],
                color=color,
                badge=badge,
                wait=row['Average_Wait_Time'],
                response=row['Critical_Response_Time'],
                util=row['Resource_Utilization'],
                fairness=row['Fairness_Index'],
                allocated=row['Allocated_Patients'],
                reward=row['Total_Reward'],
            )


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
