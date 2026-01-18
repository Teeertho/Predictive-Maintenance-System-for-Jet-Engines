import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# s4  = T50 (LPT Outlet Temperature)
# s11 = Ps30 (HPC Outlet Pressure)
MY_CHOSEN_SENSORS = ['s4', 's11'] 

# KNEE CAP
MAX_CAP = 125 

print("------------------------------------------------")
print("   ✈️  JET ENGINE TRAINING (XGBOOST EDITION)")
print("------------------------------------------------")
print(f"   -> Sensors:  {MY_CHOSEN_SENSORS}")
print(f"   -> Strategy: Gradient Boosting (Sequential Correction)")
print(f"   -> Cap Limit: {MAX_CAP}")

# ==========================================
# STATION 1: LOADING & SCALING
# ==========================================
print("\n1. Loading & Normalizing...")
train_data = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None)
train_data = train_data.iloc[:, :26]
sensor_names = [f's{i}' for i in range(1, 22)]
train_data.columns = ['unit', 'time', 'os1', 'os2', 'os3'] + sensor_names

# Scale T50 and Ps30 (0-1)
scaler = MinMaxScaler()
train_data[MY_CHOSEN_SENSORS] = scaler.fit_transform(train_data[MY_CHOSEN_SENSORS])

# ==========================================
# STATION 2: FEATURE ENGINEERING (VALUE + SLOPE)
# ==========================================
print("2. Engineering Features (Mean & Slope)...")
features = []

for col in MY_CHOSEN_SENSORS:
    # A. Rolling Mean
    mean_col = f'{col}_mean'
    train_data[mean_col] = train_data.groupby('unit')[col].transform(lambda x: x.rolling(10).mean())
    features.append(mean_col)
    
    # B. Rolling Slope
    slope_col = f'{col}_slope'
    train_data[slope_col] = train_data.groupby('unit')[mean_col].diff()
    features.append(slope_col)

# Drop NaNs
train_data.dropna(inplace=True)

# ==========================================
# STATION 3: TARGET DEFINITION
# ==========================================
print("3. Creating Piecewise Target...")
max_life = train_data.groupby('unit')['time'].max().reset_index()
max_life.columns = ['unit', 'max']
train_data = train_data.merge(max_life, on='unit', how='left')

# Calculate Linear RUL & Clip
train_data['RUL'] = train_data['max'] - train_data['time']
train_data['RUL'] = train_data['RUL'].clip(upper=MAX_CAP)

# ==========================================
# STATION 4: TRAINING (XGBOOST)
# ==========================================
print("4. Training XGBoost Regressor...")

# Hyperparameters tuned for FD001:
# n_estimators=500: Enough trees to learn the curve, but not overfit.
# learning_rate=0.01: Slow and steady learning (High Precision).
# max_depth=4: Keeps trees simple to avoid memorizing noise.
model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=100
)

print("   -> Training with Early Stopping...")
# We must pass the validation set so it knows when to stop
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=False
)

print(f"   -> Stopped at {model.best_iteration} trees!")


model.fit(train_data[features], train_data['RUL'])

# ==========================================
# STATION 5: SAVING
# ==========================================
print("5. Saving Artifacts...")

# Note: We save it as 'xgb_model.pkl' to distinguish it.
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(MY_CHOSEN_SENSORS, 'good_sensors.pkl')
joblib.dump(MAX_CAP, 'rul_config.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ SUCCESS! XGBoost Model Trained.")