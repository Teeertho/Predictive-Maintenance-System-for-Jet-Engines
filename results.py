import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. SETUP & LOADING
# ==========================================
print("--- 🚀 INITIALIZING PREDICTION ENGINE ---")

try:
    # Load the trained brain
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    good_sensors = joblib.load('good_sensors.pkl') 
    TRAINED_CAP = joblib.load('rul_config.pkl')
    
    print("✅ Model artifacts loaded.")
    print(f"   -> Sensors: {good_sensors}")

except FileNotFoundError:
    print("🚨 ERROR: Missing .pkl files. Run 'train.py' first!")
    exit()

# Load Test Data
try:
    df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None)
    df = df.iloc[:, :26]
    df.columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    print(f"✅ Loaded test data: {df.shape}")
except FileNotFoundError:
    print("🚨 ERROR: 'test_FD001.txt' not found.")
    exit()

# ==========================================
# 2. BATCH PREDICTION LOOP
# ==========================================
predictions = []
unique_units = df['unit'].unique()

print(f"--- 🔄 PROCESSING {len(unique_units)} ENGINES ---")

for unit_id in unique_units:
    # 1. Isolate Engine
    engine_df = df[df['unit'] == unit_id].copy()
    
    # 2. Scale Data (Must match training!)
    engine_df[good_sensors] = scaler.transform(engine_df[good_sensors])
    
    # 3. Create Features (Mean + Slope)
    feature_cols = []
    
    for col in good_sensors:
        # A. Mean
        mean_col = f'{col}_mean'
        engine_df[mean_col] = engine_df[col].rolling(10).mean()
        
        # B. Slope
        slope_col = f'{col}_slope'
        engine_df[slope_col] = engine_df[mean_col].diff()
        
        feature_cols.append(mean_col)
        feature_cols.append(slope_col)
    
    # 4. Clean & Predict
    clean_df = engine_df.dropna()
    
    if clean_df.empty:
        # Fallback for very short engines
        final_pred = TRAINED_CAP
    else:
        # Predict on the very last available cycle
        last_row = clean_df.iloc[[-1]][feature_cols]
        final_pred = model.predict(last_row)[0]
    
    predictions.append(final_pred)

# ==========================================
# 3. SAVING FILES
# ==========================================

# FILE A: DETAILED (For You)
df_results = pd.DataFrame({
    'Unit_ID': unique_units,
    'Prediction': [int(p) for p in predictions]
})
df_results.to_csv("final_predictions_detailed.csv", index=False)

# FILE B: RMSE FORMAT (For the Judge/Calculator)
# Just the values. No header. No index.
with open("final_predictions_RMSE.txt", "w") as f:
    for p in predictions:
        f.write(f"{int(p)}\n")

print("\n========================================")
print("✅ GENERATION COMPLETE")
print("----------------------------------------")
print("📄 1. final_predictions_detailed.csv -> (Unit IDs + Values)")
print("📄 2. final_predictions_RMSE.txt     -> (Pure Values for Scoring)")
print("========================================")