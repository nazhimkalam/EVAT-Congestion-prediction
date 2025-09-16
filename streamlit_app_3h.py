import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------- Page setup --------------------------
st.set_page_config(
    page_title="EVAT — Congestion (3h)",
    page_icon="⚡",
    layout="wide",
    menu_items={"about": "EVAT Congestion Dashboard • Queueing (M/M/c) with what-if controls"}
)

st.title("⚡ EVAT — Congestion (3-hour arrivals)")
st.caption("Predict arrivals, recompute queue waits with **Erlang-C**, and test **what-if** scenarios for charger count and service speed.")

# -------------------------- Paths & defaults --------------------------
ART = Path("artifacts_premium")
PRED_PATH   = ART / "predictions_3h_with_wait_times.csv"   # from test eval cell
FUTURE_PATH = ART / "future_forecast_3h.csv"               # from future-forecast cells
SUMMARY_PATH= ART / "forecast_summary.csv"

MU_DEFAULT = 2.0   # jobs/hour/charger (~30 min)
C_DEFAULT  = 4

# -------------------------- Data loading --------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # timestamp unification
    if "ts" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"])
    else:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    # station id unification
    if "station_id" in df.columns and "stationId" not in df.columns:
        df["stationId"] = df["station_id"].astype(str)
    if "stationId" in df.columns:
        df["stationId"] = df["stationId"].astype(str)

    # unify μ and c naming
    if "mu" not in df.columns and "mu_per_hour" in df.columns:
        df["mu"] = df["mu_per_hour"]
    if "c" not in df.columns:
        df["c"] = np.nan

    # arrivals predictions (test vs future)
    # - test file: pred_arrivals_3h, lambda_hour, expected_wait_mins, rho
    # - future file: lambda_pred_3h (expected for 3h), lo80/hi80, lo95/hi95, p_wait, rho, Wq_min
    if "pred_arrivals_3h" not in df.columns and "lambda_pred_3h" in df.columns:
        df["pred_arrivals_3h"] = df["lambda_pred_3h"]
    if "lambda_hour" not in df.columns:
        if "pred_arrivals_3h" in df.columns:
            df["lambda_hour"] = np.clip(df["pred_arrivals_3h"], 0, None) / 3.0
        else:
            df["lambda_hour"] = np.nan

    # expected wait column unification
    if "expected_wait_mins" not in df.columns and "Wq_min" in df.columns:
        df["expected_wait_mins"] = df["Wq_min"]

    # ensure optional columns exist (for charts/table)
    for col in ["arrivals", "pred_arrivals_3h", "expected_wait_mins", "expected_queue_len",
                "stationId", "lo80", "hi80", "lo95", "hi95", "p_wait", "rho"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# Try to load both sources if they exist
df_pred = load_data(PRED_PATH.as_posix()) if PRED_PATH.exists() else None
df_future = load_data(FUTURE_PATH.as_posix()) if FUTURE_PATH.exists() else None
df_summary = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else None

# -------------------------- Sidebar controls --------------------------
sources = []
if df_pred is not None: sources.append("Test predictions")
if df_future is not None: sources.append("Future forecast")
if not sources:
    st.error("No artifacts found. Run the notebook to generate files under 'artifacts_premium/'.")
    st.stop()

st.sidebar.header("Data source")
source = st.sidebar.radio("Choose", sources, index=0)

df = df_pred if source == "Test predictions" else df_future

st.sidebar.header("Filters")
stations = sorted([str(s) for s in df["stationId"].dropna().unique().tolist()])
sid = st.sidebar.selectbox("Station", stations)

# Station slice & date range
d_base = df[df["stationId"].astype(str) == sid].copy()
if d_base.empty:
    st.warning("No rows for the selected station in this source.")
    st.stop()

d_base = d_base.sort_values("timestamp")
min_dt, max_dt = d_base["timestamp"].min(), d_base["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_dt.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date()
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    d_base = d_base[(d_base["timestamp"] >= start_dt) & (d_base["timestamp"] < end_dt)]

st.sidebar.header("What-if settings")
c_multiplier  = st.sidebar.slider("Charger multiplier (c×)", 0.5, 3.0, 1.0, 0.1)
mu_multiplier = st.sidebar.slider("Service speed multiplier (μ×)", 0.5, 2.0, 1.0, 0.1)
roll_window   = st.sidebar.slider("Smoothing (hours)", 0, 6, 2, 1)
wait_target   = st.sidebar.number_input("Target max wait (mins)", min_value=0, value=15, step=5)

# -------------------------- Erlang-C helper --------------------------
def erlang_c_wait_time(lam_hour: float, mu_hour: float, c: int):
    """Return (Wq_hours, Lq, rho). Safe for edge cases."""
    if c <= 0 or mu_hour is None or (isinstance(mu_hour, float) and np.isnan(mu_hour)) or mu_hour <= 0:
        return np.nan, np.nan, np.nan
    if lam_hour <= 0:
        return 0.0, 0.0, 0.0
    rho = lam_hour / (c * mu_hour)
    if rho >= 1.0:
        return float("inf"), float("inf"), float(rho)

    a = lam_hour / mu_hour  # offered load (Erlangs)
    # sum_{n=0}^{c-1} a^n / n! computed stably
    sum_terms = 1.0
    term = 1.0
    for n in range(1, c):
        term *= a / n
        sum_terms += term
    term_c = term * (a / c) if c > 0 else 0.0

    P0 = 1.0 / (sum_terms + (term_c / (1.0 - rho)))
    Lq = (P0 * term_c * rho) / ((1.0 - rho) ** 2)
    Wq_hours = Lq / lam_hour
    return float(Wq_hours), float(Lq), float(rho)

# -------------------------- What-if recompute --------------------------
d = d_base.copy()

# Fill μ and c if absent (common for the future-forecast artifact)
if "mu" not in d.columns or d["mu"].isna().all():
    d["mu"] = MU_DEFAULT
if "c" not in d.columns or d["c"].isna().all():
    d["c"] = C_DEFAULT

d["c_adj"]  = np.maximum(1, np.round(d["c"] * c_multiplier).astype(int))
d["mu_adj"] = d["mu"] * mu_multiplier

Wq_mins_adj, rho_adj = [], []
lam_series = d.get("lambda_hour", pd.Series([np.nan] * len(d)))
for lam, mu, cc in zip(lam_series, d["mu_adj"], d["c_adj"]):
    Wq_h, Lq, rho = erlang_c_wait_time(
        float(lam) if pd.notnull(lam) else 0.0,
        float(mu) if pd.notnull(mu) else np.nan,
        int(cc)
    )
    Wq_mins_adj.append(np.nan if np.isinf(Wq_h) else Wq_h * 60.0)
    rho_adj.append(rho)

d["expected_wait_mins_adj"] = Wq_mins_adj
d["rho_adj"] = rho_adj

# Optional smoothing (visual only)
if roll_window and roll_window > 0:
    d = d.sort_values("timestamp")
    for col in ["arrivals", "pred_arrivals_3h", "expected_wait_mins", "expected_wait_mins_adj"]:
        if col in d.columns:
            d[col] = d[col].rolling(int(roll_window), min_periods=1).mean()

# -------------------------- KPIs --------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
latest = d.iloc[-1] if len(d) else pd.Series(dtype=float)

def fmt(v, nd=1):
    return "N/A" if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))) else f"{v:.{nd}f}"

latest_wait = latest.get("expected_wait_mins_adj", np.nan)
if (pd.isna(latest_wait) or latest_wait is None) and "expected_wait_mins" in d.columns:
    latest_wait = latest.get("expected_wait_mins", np.nan)
latest_q = latest.get("expected_queue_len", np.nan)
latest_rho = latest.get("rho_adj", np.nan)

with kpi_col1:
    st.metric("Latest predicted wait (mins)", fmt(latest_wait))
with kpi_col2:
    st.metric("Latest predicted queue length", fmt(latest_q, 2))
with kpi_col3:
    st.metric("Utilization ρ (latest)", fmt(latest_rho, 2))
with kpi_col4:
    tgt = int(wait_target)
    status = "N/A" if pd.isna(latest_wait) else ("✅ On-target" if latest_wait <= tgt else "⚠️ Above target")
    st.metric("Wait status vs target", status)

if pd.Series(d["rho_adj"]).dropna().ge(1.0).any():
    st.error("System enters **unstable** region (ρ≥1) in the selected window. Increase **c×** or **μ×**.", icon="🔥")

# -------------------------- Tabs --------------------------
tabs = ["Overview", "Time series", "Table"]
if df_summary is not None and source == "Future forecast":
    tabs.insert(0, "Summary")
tab_objs = st.tabs(tabs)

# ------ Optional Summary (future forecast KPIs) ------
if "Summary" in tabs:
    with tab_objs[0]:
        st.subheader("Stations at risk (Future forecast)")
        st.dataframe(df_summary.sort_values("p_wait_peak", ascending=False), use_container_width=True)

# Determine tab indices based on presence of Summary
tab_offset = 1 if "Summary" in tabs else 0
tab_overview = tab_objs[0 + tab_offset]
tab_timeser  = tab_objs[1 + tab_offset]
tab_table    = tab_objs[2 + tab_offset]

# ------ Overview: snapshot ------
with tab_overview:
    left, right = st.columns((1.2, 1), vertical_alignment="top")
    with left:
        st.subheader("Arrivals (actual vs predicted, 3h bins)")
        plot_df = d[["timestamp", "pred_arrivals_3h", "arrivals"]].melt("timestamp", var_name="series", value_name="value")
        chart1 = (
            alt.Chart(plot_df.dropna(subset=["value"]))
            .mark_line(point=False)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("value:Q", title="Arrivals (3h)"),
                color=alt.Color("series:N", title="Series",
                                scale=alt.Scale(domain=["arrivals","pred_arrivals_3h"], range=["#4C78A8","#F58518"])),
                tooltip=["timestamp:T","series:N","value:Q"]
            )
            .properties(height=280)
        )
        # Uncertainty bands if present (future mode)
        if {"lo80","hi80"}.issubset(d.columns) and d["lo80"].notna().any():
            band80 = alt.Chart(d[["timestamp","lo80","hi80"]].dropna()).mark_area(opacity=0.15).encode(
                x="timestamp:T", y="lo80:Q", y2="hi80:Q"
            )
            chart1 = band80 + chart1
        if {"lo95","hi95"}.issubset(d.columns) and d["lo95"].notna().any():
            band95 = alt.Chart(d[["timestamp","lo95","hi95"]].dropna()).mark_area(opacity=0.10).encode(
                x="timestamp:T", y="lo95:Q", y2="hi95:Q"
            )
            chart1 = band95 + chart1
        st.altair_chart(chart1, use_container_width=True)

    with right:
        st.subheader("Expected wait (minutes)")
        wait_col = "expected_wait_mins_adj" if "expected_wait_mins_adj" in d.columns else "expected_wait_mins"
        wait_df = d[["timestamp", wait_col]].rename(columns={wait_col: "wait_mins"}).dropna()
        rule = alt.Chart(pd.DataFrame({"y": [wait_target]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
        chart2 = (
            alt.Chart(wait_df)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("wait_mins:Q", title="Expected wait (mins)"),
                tooltip=["timestamp:T","wait_mins:Q"]
            )
            .properties(height=280)
        )
        st.altair_chart(chart2 + rule, use_container_width=True)

    st.info(
        "Tip: If waits exceed target, try **increasing c×** (more chargers) or **increasing μ×** (faster service). "
        "Watch utilization ρ; when ρ ≥ 1 the queue grows without bound.",
        icon="💡"
    )

# ------ Time series: richer exploration ------
with tab_timeser:
    st.subheader("Detailed time series")
    sub_left, sub_right = st.columns(2)

    with sub_left:
        st.markdown("**Arrivals (overlay)**")
        st.altair_chart(chart1.interactive(), use_container_width=True)

    with sub_right:
        st.markdown("**Wait vs target**")
        st.altair_chart((chart2 + rule).interactive(), use_container_width=True)

    # Utilization over time
    util_df = d[["timestamp", "rho_adj"]].rename(columns={"rho_adj": "rho"}).dropna()
    util_rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#D62728", strokeDash=[6,3]).encode(y="y:Q")
    util_chart = (
        alt.Chart(util_df)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("rho:Q", title="Utilization ρ"),
            tooltip=["timestamp:T","rho:Q"]
        )
        .properties(height=240)
    )
    st.markdown("**Utilization (ρ) — stability check**")
    st.altair_chart(util_chart + util_rule, use_container_width=True)

# ------ Table: export & inspect ------
with tab_table:
    st.subheader("Data (filtered)")
    show_cols = ["timestamp","stationId","arrivals","pred_arrivals_3h","lambda_hour","c","c_adj","mu","mu_adj",
                 "expected_wait_mins","expected_wait_mins_adj","expected_queue_len","rho","rho_adj",
                 "lo80","hi80","lo95","hi95","p_wait"]
    present_cols = [c for c in show_cols if c in d.columns]
    st.dataframe(d[present_cols].reset_index(drop=True), use_container_width=True, height=380)

    csv = d[present_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name=f"evat_{sid}_filtered.csv", mime="text/csv")

# ------ About ------
with st.expander("About this dashboard", expanded=False):
    st.markdown("""
**What this shows**
- **Arrivals (3h bins):** actual vs predicted counts (with **80/95% bands** when available).
- **Expected wait:** recomputed via **Erlang-C** under your what-if settings.
- **Utilization (ρ):** stability indicator (if **ρ ≥ 1**, queue diverges).

**Model**
- λ per 3h from LSTM → λ per hour = λ/3 for queueing.
- μ per hour per charger (from data or default) and **c** chargers drive **Wq** and **ρ**.
""")
