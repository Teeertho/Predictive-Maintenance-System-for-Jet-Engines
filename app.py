import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Predictive Maintenance System", layout="wide")

# --- 2. LOADER ---
@st.cache_resource
def load_system():
    try:
        model = joblib.load('rf_model.pkl')
        good_sensors = joblib.load('good_sensors.pkl')
        scaler = joblib.load('scaler.pkl')
        TRAINED_CAP = joblib.load('rul_config.pkl')
    except FileNotFoundError:
        st.error("🚨 Artifacts missing. Run 'train.py' first!")
        st.stop()
    
    # Load Test Data
    try:
        df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None)
        df = df.iloc[:, :26]
        df.columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    except:
        st.error("🚨 Missing 'test_FD001.txt'")
        st.stop()
    
    return model, good_sensors, scaler, TRAINED_CAP, df

model, good_sensors, scaler, TRAINED_CAP, df = load_system()

# --- 3. SIDEBAR ---
st.sidebar.header("✈️ Engine Selector")
unit_id = st.sidebar.selectbox("Unit ID", df['unit'].unique())
st.sidebar.markdown("---")
st.sidebar.caption(f"Mode: Piecewise Cap @ {TRAINED_CAP}")

# --- 4. PROCESSING ---
engine_df = df[df['unit'] == unit_id].copy()

# A. Scale Data
engine_df[good_sensors] = scaler.transform(engine_df[good_sensors])

# B. Create Features (Mean + Slope)
feature_cols = []
slopes = {} 

for col in good_sensors:
    mean_col = f'{col}_mean'
    engine_df[mean_col] = engine_df[col].rolling(10).mean()
    slope_col = f'{col}_slope'
    engine_df[slope_col] = engine_df[mean_col].diff()
    
    feature_cols.append(mean_col)
    feature_cols.append(slope_col)
    
    slopes[col] = engine_df[slope_col].iloc[-1]

clean_df = engine_df.dropna().copy()
if clean_df.empty:
    st.warning("⚠️ Analyzing... (Need > 10 cycles)")
    st.stop()

# --- 5. CALCULATIONS ---
ai_rul = model.predict(clean_df[feature_cols])[-1]
current_age = clean_df['time'].iloc[-1]
FLEET_AVG = 206
standard_rul = FLEET_AVG - current_age
health_pct = (ai_rul / TRAINED_CAP) * 100

# Calculate Delta (AI vs Standard)
diff = ai_rul - standard_rul

# --- 6. DASHBOARD METRICS ---
st.title(f"Engine #{unit_id} Diagnostics")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("1️⃣ Current Age", f"{int(current_age)} Cycles")

with col2:
    # --- AI PREDICTION (Standard Streamlit Metric) ---
    # We use the 'delta' parameter to color it green if it's better than standard
    label = "2️⃣ AI Prediction (Smart)"
    
    if ai_rul >= (TRAINED_CAP - 2):
        st.metric(label, f"≥ {int(ai_rul)} Cycles", delta="Max Health", delta_color="normal")
    else:
        st.metric(label, f"{int(ai_rul)} Cycles", delta=f"{int(diff)} vs Std", delta_color="normal")

with col3:
    # Standard RUL (Kept simple)
    st.metric("3️⃣ Standard RUL", f"{int(standard_rul)} Cycles", help="Manual Calculation based on Fleet Avg")

with col4:
    # Health Status (Kept colored for visibility)
    if health_pct > 70: c = "#2ecc71"
    elif health_pct > 30: c = "#f39c12"
    else: c = "#e74c3c"
    
    st.markdown(f"""
    <div style="margin-top: -5px;">
        <p style="font-size: 14px; margin-bottom: 5px; color: #888;">4️⃣ Health Status</p>
        <p style="font-size: 34px; font-weight: bold; margin: 0px; color: {c};">
            {health_pct:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. GRAPH ---
st.markdown("---")
st.subheader("📉 RUL Trajectory Comparison")

# Prepare Data
clean_df['Standard Calculation'] = FLEET_AVG - clean_df['time']
clean_df['AI Prediction'] = model.predict(clean_df[feature_cols])

# Plot
chart_data = clean_df.set_index('time')[['Standard Calculation', 'AI Prediction']]
st.line_chart(chart_data, color=["#FF0000", "#0000FF"])
st.caption("Red = Standard (Manual) | Blue = AI Prediction (Smart)")
