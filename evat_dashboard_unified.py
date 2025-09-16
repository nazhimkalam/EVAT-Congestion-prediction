
# evat_dashboard_unified.py
# Unified app with Forecast & What-If + Historical tabs
# Includes: calendar date pickers, KPIs, utilization, P(wait), Wq, worst windows table,
# capacity planner, scenario CSV download.

import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="EVAT - Unified Congestion Dashboard", page_icon="EV", layout="wide")

# ------------------ Data loaders ------------------
@st.cache_data
def load_forecast():
    df = pd.read_csv("forecast_results.csv", parse_dates=["bin_time"])
    need = {"station_id", "bin_time", "lambda_forecast", "lambda_lower", "lambda_upper"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in forecast_results.csv: {sorted(missing)}")
    return df

@st.cache_data
def load_history_optional():
    try:
        hist = pd.read_csv("history_binned.csv", parse_dates=["bin_time"])
        if {"station_id", "bin_time", "arrivals"}.issubset(hist.columns):
            return hist
    except Exception:
        pass
    return None

# ------------------ Queueing utils ------------------
def erlang_c_probability_wait(lmbda, mu, c):
    if lmbda <= 0 or mu <= 0 or c <= 0:
        return 0.0
    rho = lmbda / (c * mu)
    if rho >= 1.0:
        return 1.0
    sum_terms = sum([(lmbda/mu)**n / math.factorial(n) for n in range(c)])
    last_term = ((lmbda/mu)**c) / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (sum_terms + last_term)
    Pw = last_term * P0
    return float(min(max(Pw, 0.0), 1.0))

def mmc_metrics(lmbda, mu, c):
    if lmbda <= 0 or mu <= 0 or c <= 0:
        return 0.0, 0.0, 0.0, 0.0
    rho = lmbda / (c * mu)
    if rho >= 1.0:
        return rho, 1.0, float("inf"), float("inf")
    Pw = erlang_c_probability_wait(lmbda, mu, c)
    Lq = Pw * rho / (1 - rho)
    Wq = Lq / lmbda if lmbda > 0 else 0.0
    return float(rho), float(Pw), float(Lq), float(Wq)

def compute_queueing(df, mu, c, demand_scale=1.0):
    sub = df.copy()
    sub["lambda_scaled"] = sub["lambda_forecast"] * demand_scale
    vals = np.array([mmc_metrics(x, mu, c) for x in sub["lambda_scaled"].values])
    sub["rho"] = vals[:, 0]
    sub["p_wait"] = vals[:, 1]
    sub["Lq"] = vals[:, 2]
    sub["Wq_minutes"] = vals[:, 3] * 60.0
    return sub

def required_c_for_point(lmbda, mu, sla_minutes, c_max=100):
    if lmbda <= 0 or mu <= 0:
        return 1
    for c in range(1, c_max + 1):
        rho, Pw, Lq, Wq = mmc_metrics(lmbda, mu, c)
        if np.isfinite(Wq) and (Wq * 60.0) <= sla_minutes:
            return c
    return c_max

def capacity_planner(df, mu, sla_minutes, coverage=0.9):
    req_cs = [required_c_for_point(x, mu, sla_minutes) for x in df["lambda_scaled"].values]
    req_series = pd.Series(req_cs).sort_values()
    idx = int(np.ceil(coverage * len(req_series))) - 1
    idx = max(0, min(idx, len(req_series) - 1))
    return int(req_series.iloc[idx]), req_series

# ------------------ App ------------------
fc = load_forecast()
hist = load_history_optional()

stations = sorted(fc["station_id"].unique().tolist())
st.title("EVAT - Unified Congestion Dashboard")

tab_forecast, tab_history = st.tabs(["Forecast & What-If", "Historical (optional)"])

# =====================================================================
# Forecast & What-If (with calendar)
# =====================================================================
with tab_forecast:
    # Station + time selection mode
    top1, top2 = st.columns([1.2, 2.0])
    station = top1.selectbox("Station", stations, index=0)

    fc_station = fc[fc["station_id"] == station].sort_values("bin_time").copy()
    min_dt = fc_station["bin_time"].min().date()
    max_dt = fc_station["bin_time"].max().date()

    time_mode = top2.radio(
        "Time selection",
        ["Horizon (steps)", "Date range"],
        horizontal=True,
        index=0,
    )

    if time_mode == "Horizon (steps)":
        max_h = fc_station.shape[0]
        horizon = st.slider(
            "Horizon (future steps)",
            min_value=8,
            max_value=max_h,
            value=min(56, max_h),
            step=4,
        )
        sub_base = fc_station.head(horizon).copy()
    else:
        picked = st.date_input(
            "Pick forecast date range",
            value=(min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
            key="fc_dates",
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            start_date, end_date = picked
        else:
            start_date = end_date = picked
        start_ts = pd.Timestamp(start_date)
        end_ts_inc = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        sub_base = fc_station[(fc_station["bin_time"] >= start_ts) &
                              (fc_station["bin_time"] < end_ts_inc)].copy()
        if sub_base.empty:
            st.warning("No forecast bins in the selected date range.")
            st.stop()

    # Other controls
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.0, 1.2, 1.2])
    mu_mode = c3.radio("Service rate input", ["mu (per hr)", "Avg session (mins)"], horizontal=True)
    if mu_mode == "mu (per hr)":
        mu = c4.number_input("mu per server per hour", min_value=0.1, max_value=20.0, value=2.0, step=0.1, format="%.1f")
    else:
        sess = c4.number_input("Avg session time (mins)", min_value=5, max_value=240, value=30, step=5)
        mu = 60.0 / sess

    c_servers = c5.number_input("Servers (chargers, c)", min_value=1, max_value=100, value=4, step=1)

    c6, c7, c8 = st.columns([1.2, 1.2, 1.2])
    sla = c6.number_input("Target wait SLA (minutes)", min_value=1, max_value=180, value=15, step=1)
    demand_scale = c7.slider("Demand scaling (what-if)", 50, 150, 100, 5) / 100.0
    coverage = c8.slider("Capacity planner coverage (%)", 50, 99, 90, 1) / 100.0

    # Compute queueing on chosen period
    sub = compute_queueing(sub_base, mu=mu, c=int(c_servers), demand_scale=demand_scale)

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    avg_lambda = float(sub["lambda_scaled"].mean())
    peak_lambda = float(sub["lambda_scaled"].max())
    avg_rho = float(np.clip(sub["rho"], 0, 5).mean())
    max_rho = float(np.clip(sub["rho"], 0, 5).max())
    pct_wait_high = float((sub["p_wait"] > 0.5).mean() * 100)
    pct_sla_breach = float((sub["Wq_minutes"] > sla).mean() * 100)
    k1.metric("Avg lambda (per 3h)", f"{avg_lambda:.1f}")
    k2.metric("Peak lambda (per 3h)", f"{peak_lambda:.1f}")
    k3.metric("Avg utilization rho", f"{avg_rho:.2f}")
    k4.metric("Max utilization rho", f"{max_rho:.2f}")
    k5.metric("% time P(wait) > 0.5", f"{pct_wait_high:.0f}%")
    k6.metric(f"% time Wq > SLA({sla}m)", f"{pct_sla_breach:.0f}%")

    # Charts
    st.markdown("### Lambda forecast with 80% interval")
    band = alt.Chart(sub).mark_area(opacity=0.2).encode(x="bin_time:T", y="lambda_lower:Q", y2="lambda_upper:Q")
    line = alt.Chart(sub.assign(lambda_scaled=sub["lambda_scaled"])).mark_line().encode(x="bin_time:T", y="lambda_scaled:Q")
    st.altair_chart(band + line, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Utilization (rho)")
        st.altair_chart(
            alt.Chart(sub.assign(rho_clip=sub["rho"].clip(upper=2.0)))
               .mark_line()
               .encode(x="bin_time:T", y="rho_clip:Q"),
            use_container_width=True
        )
        st.caption("Note: rho >= 1 means unstable; queues grow quickly.")
    with right:
        st.markdown("### Probability of waiting")
        st.altair_chart(alt.Chart(sub).mark_line().encode(x="bin_time:T", y="p_wait:Q"), use_container_width=True)

    st.markdown("### Expected wait (minutes)")
    wait_chart = alt.Chart(sub.assign(Wq_cap=sub["Wq_minutes"].clip(upper=120))).mark_line().encode(x="bin_time:T", y="Wq_cap:Q")
    sla_rule = alt.Chart(pd.DataFrame({'y': [sla]})).mark_rule(strokeDash=[6, 4]).encode(y='y:Q')
    st.altair_chart(wait_chart + sla_rule, use_container_width=True)

    # Worst windows
    st.markdown("### Worst windows (by expected wait)")
    worst = (sub.sort_values("Wq_minutes", ascending=False)
             [["bin_time", "lambda_scaled", "rho", "p_wait", "Wq_minutes"]]
             .head(12))
    worst = worst.rename(columns={"lambda_scaled": "lambda(3h)", "rho": "rho",
                                  "p_wait": "P(wait)", "Wq_minutes": "Wq(min)"})
    st.dataframe(worst, use_container_width=True)

    # Capacity planner
    req_c, req_series = capacity_planner(sub, mu=mu, sla_minutes=sla, coverage=coverage)
    colA, colB = st.columns([1.2, 2])
    colA.metric(f"Chargers needed for SLA<={sla}m at {int(coverage*100)}% coverage",
                f"{req_c}", delta=f"{int(req_c) - int(c_servers)} vs current")
    colB.altair_chart(
        alt.Chart(pd.DataFrame({"required_c": req_series}))
           .transform_bin("bin", "required_c", bin=alt.Bin(maxbins=20))
           .mark_bar()
           .encode(x="bin:N", y="count()"),
        use_container_width=True
    )

    # Download scenario snapshot
    st.download_button(
        "Download current scenario (CSV)",
        data=sub.to_csv(index=False).encode("utf-8"),
        file_name=f"evat_scenario_{station}.csv",
        mime="text/csv"
    )

# =====================================================================
# Historical (optional) with calendar
# =====================================================================
with tab_history:
    st.subheader("Historical view (optional)")
    if hist is None:
        st.info("No history_binned.csv found. Export from the notebook with columns: station_id, bin_time, arrivals.")
    else:
        s2 = st.selectbox("Station (history)", sorted(hist["station_id"].unique().tolist()))
        dfh_all = hist[hist["station_id"] == s2].sort_values("bin_time").copy()

        h_min = dfh_all["bin_time"].min().date()
        h_max = dfh_all["bin_time"].max().date()
        h_range = st.date_input(
            "Pick history date range",
            value=(h_min, h_max),
            min_value=h_min,
            max_value=h_max,
            key="hist_dates",
        )
        if isinstance(h_range, tuple) and len(h_range) == 2:
            h_start, h_end = h_range
        else:
            h_start, h_end = h_range, h_range

        h_start_ts = pd.Timestamp(h_start)
        h_end_ts_inc = pd.Timestamp(h_end) + pd.Timedelta(days=1)
        dfh = dfh_all[(dfh_all["bin_time"] >= h_start_ts) & (dfh_all["bin_time"] < h_end_ts_inc)]

        if dfh.empty:
            st.warning("No historical data in the selected date range.")
        else:
            st.line_chart(dfh.set_index("bin_time")["arrivals"], use_container_width=True)
            st.caption("Historical arrivals per 3h.")
