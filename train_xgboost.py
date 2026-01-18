import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # <--- New Import
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
# The "Physics" Sensors: T50 (LPT Temp) and Ps30 (HPC Pressure)
MY_CHOSEN_SENSORS = ['s4', 's11'] 
MAX_CAP = 125  # The "Knee" Point

print(f"Initializing Physics-Based Training...")
print(f"Sensors: {MY_CHOSEN_SENSORS}")
print(f"RUL Cap: {MAX_CAP}")

# ==========================================
# STEP 1: LOAD & PREPARE TRAINING DATA
# ==========================================
print("\n--- Step 1: Loading Training Data ---")
columns = ['unit', 'time', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=columns)

# 1. Scale the raw sensors (Normalization)
print("   -> Normalizing sensors...")
scaler = MinMaxScaler()
train_df[MY_CHOSEN_SENSORS] = scaler.fit_transform(train_df[MY_CHOSEN_SENSORS])

# 2. Feature Engineering (Velocity & Smoothing)
print("   -> Calculating Velocity (Slopes) and Smoothing...")
features = []
for col in MY_CHOSEN_SENSORS:
    # A. Rolling Mean (The "Anti-Shake" Glasses)
    mean_col = f'{col}_mean'
    train_df[mean_col] = train_df.groupby('unit')[col].transform(lambda x: x.rolling(10).mean())
    features.append(mean_col)
    
    # B. Slope (The "Speedometer")
    slope_col = f'{col}_slope'
    train_df[slope_col] = train_df.groupby('unit')[mean_col].diff().fillna(0)
    features.append(slope_col)

# Drop early rows where rolling mean is NaN
train_df.dropna(inplace=True)

# ==========================================
# STEP 2: CALCULATE RUL (TARGET)
# ==========================================
print("\n--- Step 2: Defining the Target (RUL) ---")
# Find the maximum life of each engine
max_life = train_df.groupby('unit')['time'].max().reset_index()
max_life.columns = ['unit', 'max']

# Merge back to main table
train_df = train_df.merge(max_life, on='unit', how='left')

# Calculate Linear RUL
train_df['RUL'] = train_df['max'] - train_df['time']

# Apply the "Knee" Cap (Piecewise Linear)
print(f"   -> Applying Cap at {MAX_CAP} cycles...")
train_df['RUL'] = train_df['RUL'].clip(upper=MAX_CAP)

# ==========================================
# STEP 3: TRAIN MODEL (UPDATED FOR M4 AIR)
# ==========================================
print("\n--- Step 3: Training XGBoost with Early Stopping ---")

X = train_df[features]
y = train_df['RUL']

# 1. Create an internal validation set (20%) 
# This lets the model test itself while learning to know when to stop.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Configure Model
model = XGBRegressor(
    n_estimators=3000,          # High ceiling (M4 handles this easily)
    learning_rate=0.01,         # Slow & Precise learning
    max_depth=4,                # Shallow trees to prevent overfitting
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,                  # Use all M4 cores
    random_state=42,
    early_stopping_rounds=100    # STOP if no improvement for 100 rounds
)

# 3. Fit with Watchdog
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=False
)

print(f"   -> Optimization finished! Best iteration: {model.best_iteration}")

# Save the model
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl') # Save scaler to reuse on test data
print("   -> Model and Scaler saved.")

# ==========================================
# STEP 4: GENERATE FINAL PREDICTIONS
# ==========================================
print("\n--- Step 4: Generating Submission File ---")
test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=columns)

predictions = []
unique_units = test_df['unit'].unique()

for unit_id in unique_units:
    # Get data for one engine
    engine_df = test_df[test_df['unit'] == unit_id].copy()
    
    # 1. Apply SAME Scaling
    engine_df[MY_CHOSEN_SENSORS] = scaler.transform(engine_df[MY_CHOSEN_SENSORS])
    
    # 2. Apply SAME Feature Engineering
    for col in MY_CHOSEN_SENSORS:
        mean_col = f'{col}_mean'
        slope_col = f'{col}_slope'
        
        # Calculate Rolling Mean & Slope
        engine_df[mean_col] = engine_df[col].rolling(10).mean()
        engine_df[slope_col] = engine_df[mean_col].diff().fillna(0)

    # 3. Select the LAST row (Current State)
    # We use double brackets [[...]] to keep it as a DataFrame
    last_row = engine_df.iloc[[-1]][features]
    
    # 4. Predict
    pred = model.predict(last_row)[0]
    predictions.append(pred)

# Write to file
output_file = "final_predictions_RMSE.txt"
with open(output_file, "w") as f:
    for p in predictions:
        f.write(f"{int(p)}\n")

print(f"   -> Done! Predictions saved to {output_file}")
