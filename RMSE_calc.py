import numpy as np
from sklearn.metrics import mean_squared_error
import os

print("--- 📊 SCORING SYSTEM ---")

# 1. Check if Answer Key exists and has data
if not os.path.exists('real_results.txt'):
    print("🚨 ERROR: 'real_results.txt' is missing!")
    print("👉 Please copy this file from the NASA dataset folder.")
    exit()

if os.path.getsize('real_results.txt') == 0:
    print("🚨 ERROR: 'real_results.txt' is EMPTY!")
    print("👉 You likely created a blank file. You need the real data from NASA.")
    exit()

# 2. Load Data
try:
    y_pred = np.loadtxt('final_predictions_RMSE.txt')
    y_true = np.loadtxt('real_results.txt')
except Exception as e:
    print(f"🚨 Read Error: {e}")
    exit()

# 3. Validation
print(f"Loaded {len(y_pred)} predictions.")
print(f"Loaded {len(y_true)} answers.")

if len(y_pred) != len(y_true):
    print("🚨 LENGTH MISMATCH!")
    print("Check if you are using the correct test file (FD001).")
    exit()

# 4. Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("---------------------------")
print(f"✅ FINAL RMSE SCORE: {rmse:.2f}")
print("---------------------------")
if rmse < 25:
    print("🏆 EXCELLENT! (Judge Standard)")
elif rmse < 40:
    print("👍 GOOD (Passing)")
else:
    print("⚠️ NEEDS IMPROVEMENT")
