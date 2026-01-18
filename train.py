import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# s4  = T50 (LPT Outlet Temperature) -> Rises with wear
# s11 = Ps30 (HPC Outlet Pressure)   -> Drops with wear
MY_CHOSEN_SENSORS = ['s4', 's11'] 

# THE KNEE CAP
# The model will learn to hold RUL at 125 until the sensors 
# show the "velocity spike" indicating degradation.
MAX_CAP = 125 

print("------------------------------------------------")
print("   ✈️  FINAL PRODUCTION TRAINING")
print("------------------------------------------------")
print(f"   -> Sensors:  {MY_CHOSEN_SENSORS} (T50 & Ps30)")
print(f"   -> Strategy: Piecewise RUL (Cap @ {MAX_CAP})")
print(f"   -> Logic:    Value + Velocity (Slope) Detection")
print(f"   -> Accuracy: 1000 Trees")

# ==========================================
# STATION 1: LOADING & SCALING
# ==========================================
print("\n1. Loading & Normalizing...")
train_data = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None)
train_data = train_data.iloc[:, :26]
sensor_names = [f's{i}' for i in range(1, 22)]
train_data.columns = ['unit', 'time', 'os1', 'os2', 'os3'] + sensor_names

# Scale T50 and Ps30 (0-1) to standardize the inputs
scaler = MinMaxScaler()
train_data[MY_CHOSEN_SENSORS] = scaler.fit_transform(train_data[MY_CHOSEN_SENSORS])

# ==========================================
# STATION 2: FEATURE ENGINEERING (VALUE + SLOPE)
# ==========================================
print("2. Engineering Features (Mean & Slope)...")
features = []

for col in MY_CHOSEN_SENSORS:
    # A. Rolling Mean (Smooths the noise)
    mean_col = f'{col}_mean'
    train_data[mean_col] = train_data.groupby('unit')[col].transform(lambda x: x.rolling(10).mean())
    features.append(mean_col)
    
    # B. Rolling Slope (The "Velocity" of failure)
    # This detects if the sensor is dipping/rising too fast
    slope_col = f'{col}_slope'
    train_data[slope_col] = train_data.groupby('unit')[mean_col].diff()
    features.append(slope_col)

# Drop NaNs created by rolling/diff (First 10 cycles)
train_data.dropna(inplace=True)

# ==========================================
# STATION 3: TARGET DEFINITION
# ==========================================
print("3. Creating Piecewise 'Knee' Target...")
max_life = train_data.groupby('unit')['time'].max().reset_index()
max_life.columns = ['unit', 'max']
train_data = train_data.merge(max_life, on='unit', how='left')

# Calculate Linear RUL
train_data['RUL'] = train_data['max'] - train_data['time']

# Apply the Cap (The Knee)
train_data['RUL'] = train_data['RUL'].clip(upper=MAX_CAP)

# ==========================================
# STATION 4: TRAINING
# ==========================================
print("4. Training Random Forest (1000 Estimators)...")
model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)
model.fit(train_data[features], train_data['RUL'])

# ==========================================
# STATION 5: SAVING
# ==========================================
print("5. Saving Artifacts...")
joblib.dump(model, 'rf_model.pkl')
joblib.dump(MY_CHOSEN_SENSORS, 'good_sensors.pkl')
joblib.dump(MAX_CAP, 'rul_config.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ SUCCESS! System is ready for the dashboard.")