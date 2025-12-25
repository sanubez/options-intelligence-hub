import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import plotly.graph_objects as go

# ======================================================
# BASIC CONFIG (FAST LOAD)
# ======================================================

st.set_page_config(
    page_title="Options Intelligence Hub",
    layout="wide"
)

IST = pytz.timezone("Asia/Kolkata")
BASE_URL = "https://api.india.delta.exchange"

# ======================================================
# SIMPLE HEADER (RENDERS IMMEDIATELY)
# ======================================================

st.title("🎯 Options Intelligence Hub")
st.caption("Decision-Support Dashboard | Cloud-Safe Version")

st.divider()

# ======================================================
# UTILITY
# ======================================================

def ist_now():
    return datetime.now(IST)

# ======================================================
# API FUNCTIONS (CACHED)
# ======================================================

@st.cache_data(ttl=30)
def get_spot(symbol="BTCUSD"):
    r = requests.get(f"{BASE_URL}/v2/tickers/{symbol}", timeout=10)
    if r.status_code == 200:
        return float(r.json()["result"]["spot_price"])
    return None


@st.cache_data(ttl=300)
def get_expiries():
    r = requests.get(f"{BASE_URL}/v2/products", timeout=10)
    expiries = set()
    if r.status_code == 200:
        for p in r.json().get("result", []):
            if p.get("contract_type") == "call_options":
                t = p.get("settlement_time")
                if t:
                    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
                    expiries.add(dt.strftime("%d-%m-%Y"))
    return sorted(list(expiries))


@st.cache_data(ttl=60)
def get_options_chain(asset, expiry):
    params = {
        "contract_types": "call_options,put_options",
        "underlying_asset_symbols": asset
    }
    r = requests.get(f"{BASE_URL}/v2/tickers", params=params, timeout=10)
    if r.status_code == 200:
        return r.json().get("result", [])
    return []


# ======================================================
# SIDEBAR (NO API CALLS AT LOAD)
# ======================================================

with st.sidebar:
    st.subheader("⚙️ Controls")

    asset = st.selectbox("Asset", ["BTC", "ETH"])
    load_exp = st.button("📅 Load Expiries")

    if load_exp:
        st.session_state.expiries = get_expiries()

    if "expiries" in st.session_state:
        expiry = st.selectbox("Expiry", st.session_state.expiries)
    else:
        expiry = None

    run = st.button("🚀 Load Market Data")

# ======================================================
# MAIN DATA LOAD (ONLY ON BUTTON)
# ======================================================

if run and expiry:
    with st.spinner("Fetching market data..."):
        spot = get_spot(f"{asset}USD")
        chain = get_options_chain(asset, expiry)

        if not spot or not chain:
            st.error("Failed to fetch data")
        else:
            st.session_state.data = {
                "spot": spot,
                "chain": chain,
                "time": ist_now()
            }

# ======================================================
# DISPLAY DATA
# ======================================================

if "data" in st.session_state:
    data = st.session_state.data

    st.success(
        f"Data Loaded | Spot: {data['spot']:.2f} | "
        f"{data['time'].strftime('%H:%M:%S IST')}"
    )

    # --------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Spot Price", f"${data['spot']:,.2f}")

    calls = sum(float(o.get("oi", 0)) for o in data["chain"] if "call" in o["contract_type"])
    puts = sum(float(o.get("oi", 0)) for o in data["chain"] if "put" in o["contract_type"])

    with col2:
        st.metric("Call OI", f"{calls:,.0f}")

    with col3:
        st.metric("Put OI", f"{puts:,.0f}")

    # --------------------------------------------------
    # PCR
    # --------------------------------------------------

    pcr = puts / calls if calls > 0 else 0
    st.metric("PCR (OI)", f"{pcr:.2f}")

    # --------------------------------------------------
    # OI DISTRIBUTION CHART
    # --------------------------------------------------

    strikes = {}
    for o in data["chain"]:
        strike = float(o.get("strike_price", 0))
        if strike not in strikes:
            strikes[strike] = {"call": 0, "put": 0}
        if "call" in o["contract_type"]:
            strikes[strike]["call"] += float(o.get("oi", 0))
        else:
            strikes[strike]["put"] += float(o.get("oi", 0))

    df = pd.DataFrame([
        {"strike": k, "call": v["call"], "put": -v["put"]}
        for k, v in strikes.items()
    ]).sort_values("strike")

    fig = go.Figure()
    fig.add_bar(x=df["strike"], y=df["call"], name="Call OI")
    fig.add_bar(x=df["strike"], y=df["put"], name="Put OI")

    fig.add_vline(x=data["spot"], line_dash="dash", line_color="yellow")

    fig.update_layout(
        barmode="relative",
        height=500,
        title="Open Interest Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Select asset, load expiry, then click **Load Market Data**")
