import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EVAT Congestion (3h)", layout="wide")
st.title("EVAT — Congestion (3‑hour arrivals)")

# ---- Load predictions with queueing inputs ----
path = "predictions_3h_with_wait_times.csv"
df = pd.read_csv(path, parse_dates=["timestamp"])

# Sanity: ensure inputs exist (lambda_hour, mu, c). If missing, derive lambda_hour from 3h counts.
if "lambda_hour" not in df.columns and "pred_arrivals_3h" in df.columns:
    df["lambda_hour"] = np.clip(df["pred_arrivals_3h"], 0, None) / 3.0

# ---- Sidebar filters & what‑if controls ----
st.sidebar.header("Filters")
stations = sorted(df["stationId"].unique().tolist())
sid = st.sidebar.selectbox("Station", stations)

st.sidebar.header("What‑if settings")
c_multiplier  = st.sidebar.slider("Charger multiplier (c×)", 0.5, 3.0, 1.0, 0.1)
mu_multiplier = st.sidebar.slider("Service speed multiplier (μ×)", 0.5, 2.0, 1.0, 0.1)

# ---- Erlang‑C (same math used in the notebook) ----
def erlang_c_wait_time(lam_hour: float, mu_hour: float, c: int):
    """Return (Wq_hours, Lq, rho). Uses numerically safe products."""
    if c <= 0 or mu_hour is None or np.isnan(mu_hour) or mu_hour <= 0:
        return np.nan, np.nan, np.nan
    if lam_hour <= 0:
        return 0.0, 0.0, 0.0
    rho = lam_hour / (c * mu_hour)
    if rho >= 1.0:
        return float("inf"), float("inf"), float(rho)

    a = lam_hour / mu_hour  # traffic intensity in Erlang
    # Compute sum_{n=0}^{c-1} a^n / n! safely
    sum_terms = 0.0
    term = 1.0  # a^0 / 0! = 1
    sum_terms += term
    for n in range(1, c):
        term *= a / n
        sum_terms += term
    # term_c = a^c / c!
    term_c = term * (a / c) if c > 0 else 0.0

    P0 = 1.0 / (sum_terms + (term_c / (1.0 - rho)))
    Lq = (P0 * term_c * rho) / ((1.0 - rho) ** 2)
    Wq_hours = Lq / lam_hour
    return float(Wq_hours), float(Lq), float(rho)

# ---- Slice station & recompute what‑if wait times ----
d = df[df["stationId"] == sid].sort_values("timestamp").copy()

# Apply multipliers (round c to nearest int and minimum 1)
if "c" in d.columns:
    d["c_adj"] = np.maximum(1, np.round(d["c"] * c_multiplier).astype(int))
else:
    d["c_adj"] = 1  # fallback

if "mu" in d.columns:
    d["mu_adj"] = d["mu"] * mu_multiplier
else:
    d["mu_adj"] = np.nan  # fallback; will yield NaN waits

# Recompute wait times row-by-row
Wq_mins_adj = []
rho_adj = []
for lam, mu, cc in zip(d.get("lambda_hour", pd.Series([np.nan]*len(d))), d["mu_adj"], d["c_adj"]):
    Wq_h, Lq, rho = erlang_c_wait_time(float(lam) if pd.notnull(lam) else 0.0,
                                       float(mu) if pd.notnull(mu) else np.nan,
                                       int(cc))
    # Cap inf for display
    if np.isinf(Wq_h):
        Wq_mins_adj.append(np.nan)  # we'll format as "unstable" in metric
    else:
        Wq_mins_adj.append(Wq_h * 60.0)
    rho_adj.append(rho)

d["expected_wait_mins_adj"] = Wq_mins_adj
d["rho_adj"] = rho_adj

# ---- Layout ----
col1, col2 = st.columns(2)

with col1:
    st.write("### Arrivals (actual vs predicted, 3h bins)")
    # Some rows (earliest) may not have 'arrivals' aligned; fill with 0 for plotting
    to_plot = d.set_index("timestamp")[["pred_arrivals_3h"]].copy()
    if "arrivals" in d.columns:
        to_plot["arrivals"] = d.set_index("timestamp")["arrivals"]
        to_plot = to_plot[["arrivals", "pred_arrivals_3h"]]
    st.line_chart(to_plot)

with col2:
    st.write("### Expected wait time (minutes)")
    # Prefer adjusted waits if μ and c are present; fall back to original if not
    if "expected_wait_mins" in d.columns and d["mu"].notna().any() and d["c"].notna().any():
        st.caption("Showing **what‑if** wait times with sliders (recomputed via Erlang‑C).")
        st.line_chart(d.set_index("timestamp")[["expected_wait_mins_adj"]])
    else:
        st.caption("Base wait times (no μ/c info available to recompute).")
        st.line_chart(d.set_index("timestamp")[["expected_wait_mins"]])

# ---- KPIs ----
latest = d.iloc[-1]
# Wait metric formatting
wait_val = latest["expected_wait_mins_adj"]
if pd.isna(wait_val) and "expected_wait_mins" in d.columns:
    wait_val = latest.get("expected_wait_mins", np.nan)

wait_text = "N/A"
if pd.notnull(wait_val):
    wait_text = f"{wait_val:.1f}"
elif pd.notnull(latest.get("rho_adj", np.nan)) and latest["rho_adj"] >= 1.0:
    wait_text = "Unstable (ρ≥1)"

st.metric("Latest predicted wait (mins)", wait_text)

qlen = latest.get("expected_queue_len", np.nan)
st.metric("Latest predicted queue length", f"{qlen:.2f}" if pd.notnull(qlen) else "N/A")

# Utilization readout (helps debug stability)
rho_disp = latest.get("rho_adj", np.nan)
st.metric("Utilization ρ (latest)", f"{rho_disp:.2f}" if pd.notnull(rho_disp) else "N/A")

st.caption("Tip: increase **c×** or **μ×** if the system becomes unstable (ρ≥1).")
