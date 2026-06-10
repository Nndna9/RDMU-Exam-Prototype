# 🏥 Smart Emergency Department Resource Allocation System

Streamlit dashboard for a decision-support system that dynamically allocates beds, doctors, nurses, and ventilators in a hospital emergency department using Reinforcement Learning, MDPs, Fuzzy Logic, MCDM (AHP + TOPSIS), Utility Theory, and Linear Optimization.

---

## 📊 Dashboard Pages

| # | Page | What it shows |
|---|---|---|
| 1 | **Hospital Overview** | Live KPIs, hourly arrivals, severity distribution, resource utilization |
| 2 | **Live Simulation** | Real-time patient stream (2-sec refresh), queue, resource status |
| 3 | **Fuzzy Triage** | Membership functions, urgency distribution, patient lookup |
| 4 | **MCDM Analysis** | AHP weights, TOPSIS rankings, filterable patient priority table |
| 5 | **RL Analytics** | Reward/learning curves, policy heatmap, ε-comparison |
| 6 | **Optimization** | Strategy comparison (RL vs Opt vs Hybrid), radar chart |
| 7 | **Trade-Off Studio** | Interactive sliders, what-if simulation, sensitivity sweep |
| 8 | **Executive Dashboard** | Performance gauges, heatmaps, strategy summary cards |

---

## 🚀 Deploy to Streamlit Cloud

### 1. Push this folder to a GitHub repository
```bash
git init
git add .
git commit -m "Initial commit: ED Resource Allocation Dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### 2. Connect to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repository, branch (`main`), and main file: `app.py`
4. Click **Deploy**

Streamlit Cloud auto-detects `requirements.txt` and installs all dependencies.

---

## 💻 Run Locally

```bash
# 1. Clone or unzip
cd ed_dashboard

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate            # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📁 Project Structure

```
ed_dashboard/
├── app.py                      # Main Streamlit application (8 pages)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Theme & server config (light mode)
└── data/
    ├── hospital_data.csv       # 1,000 synthetic ED patients
    ├── hospital_fuzzy_scores.csv  # Fuzzy triage urgency scores
    ├── patient_rankings.csv    # AHP + TOPSIS priority rankings
    ├── allocation_results.csv  # Resource allocation decisions
    └── evaluation_metrics.csv  # Strategy comparison metrics
```

---

## 🎨 Design Notes

- **Color contrast** — Dark navy sidebar with light text; light main background with dark text. All charts use white plot backgrounds with dark axis labels for readability.
- **Light theme forced** via `.streamlit/config.toml` so the dashboard looks identical for every viewer regardless of their browser/OS preference.
- **Plotly charts** use a custom helper (`style_plotly`) to guarantee consistent fonts, colors, and gridlines across all pages.

---

## 🔧 Customization

- **Change theme colors** → edit `.streamlit/config.toml`
- **Adjust dashboard palette** → edit the `COLORS` dict at top of `app.py`
- **Add a new page** → write a function and add it to the `PAGES` dict at the bottom
- **Swap data** → replace CSVs in `data/` (keep column names identical)

---

## 📦 Tech Stack

- **Streamlit** — UI framework
- **Plotly** — interactive charts
- **Pandas / NumPy** — data manipulation
- **streamlit-autorefresh** — live simulation page (2-second tick)
- **SciPy** — statistical functions

---
## NOTE
Make sure you run the collab notebook, upload data from there to the Dashboard!


---
## 📜 License

For academic use. Built as part of an RDMU course project.
