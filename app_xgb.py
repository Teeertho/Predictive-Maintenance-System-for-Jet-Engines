import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Based RUL Calc", layout="wide")

# --- 2. LOADER ---
@st.cache_resource
def load_system():
    try:
        # LOAD XGBOOST MODEL HERE
        model = joblib.load('xgb_model.pkl') 
        good_sensors = joblib.load('good_sensors.pkl')
        scaler = joblib.load('scaler.pkl')
        TRAINED_CAP = joblib.load('rul_config.pkl')
    except FileNotFoundError:
        st.error("🚨 Artifacts missing. Run 'train_xgboost.py' first!")
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
st.sidebar.header("🚀 Engine Selector")
unit_id = st.sidebar.selectbox("Unit ID", df['unit'].unique())
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: XGBoost (Gradient Boosting)")
st.sidebar.caption(f"Cap: {TRAINED_CAP} Cycles")

# --- 4. PROCESSING ---
engine_df = df[df['unit'] == unit_id].copy()
original_df = engine_df.copy() # KEEP RAW DATA FOR PLOTTING LATER

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
# Predict using XGBoost
ai_rul = model.predict(clean_df[feature_cols])[-1]

current_age = clean_df['time'].iloc[-1]
FLEET_AVG = 206
standard_rul = FLEET_AVG - current_age
health_pct = (ai_rul / TRAINED_CAP) * 100

# Calculate Delta
diff = ai_rul - standard_rul

# --- 6. DASHBOARD METRICS ---
st.title(f"Engine Diagnostics | Engine #{unit_id}")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("1️⃣ Current Age", f"{int(current_age)} Cycles")

with col2:
    # --- AI PREDICTION (Big & Clean) ---
    label = "2️⃣ AI Prediction"
    
    if ai_rul >= (TRAINED_CAP - 2):
        st.metric(label, f"≥ {int(ai_rul)} Cycles", delta="Max Health", delta_color="normal")
    else:
        # Color logic for delta
        st.metric(label, f"{int(ai_rul)} Cycles", delta=f"{int(diff)} vs Std", delta_color="normal")

with col3:
    st.metric("3️⃣ Standard RUL", f"{int(standard_rul)} Cycles", help="Manual Calculation (206 - Age)")

with col4:
    # Health Status
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

# --- 7. GRAPH 1: RUL TRAJECTORY ---
st.markdown("---")
st.subheader("📉 RUL Trajectory Comparison")

# Prepare Data
clean_df['Standard Calculation'] = FLEET_AVG - clean_df['time']
clean_df['XGBoost Prediction'] = model.predict(clean_df[feature_cols])

# Plot
chart_data = clean_df.set_index('time')[['Standard Calculation', 'XGBoost Prediction']]
st.line_chart(chart_data, color=["#FF0000", "#0000FF"])
st.caption("Red = Standard (Manual) | Blue = XGBoost (Gradient Boosting)")

# --- 8. GRAPH 2: SEPARATED SENSORS ---
st.markdown("---")
st.subheader("👁️ Sensor Analysis (The 'Why')")
st.caption("Visualizing the raw thermodynamic data that drove the AI's decision.")

# Create Two Vertical Subplots
fig_sensors = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.1,
    subplot_titles=("Temperature (T50) - Increasing Trend", "Pressure (Ps30) - Decreasing Trend")
)

# Plot 1: Temperature (Red)
fig_sensors.add_trace(
    go.Scatter(
        x=original_df['time'], 
        y=original_df['s4'], 
        name="Temp (T50)", 
        line=dict(color='#d62728', width=2)
    ),
    row=1, col=1
)

# Plot 2: Pressure (Blue)
fig_sensors.add_trace(
    go.Scatter(
        x=original_df['time'], 
        y=original_df['s11'], 
        name="Pressure (Ps30)", 
        line=dict(color='#1f77b4', width=2)
    ),
    row=2, col=1
)

# Formatting
fig_sensors.update_layout(
    height=600,  # Taller layout for two graphs
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False # Hide legend since titles explain it
)

# Axis Labels
fig_sensors.update_yaxes(title_text="Temp (°R)", row=1, col=1)
fig_sensors.update_yaxes(title_text="Pressure (psia)", row=2, col=1)
fig_sensors.update_xaxes(title_text="Cycles", row=2, col=1)

st.plotly_chart(fig_sensors, use_container_width=True)
