import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
SEASON = 2025

FORM_FILE = Path("data/processed/form_features_2025.csv")
QUALI_FILE = Path("data/raw/qualifying_results_2025.csv")

# -----------------------------
# LOAD & MERGE
# -----------------------------
form = pd.read_csv(FORM_FILE)
quali = pd.read_csv(QUALI_FILE)

df = form.merge(
    quali[[
        "season", "round", "DriverNumber",
        "qualifying_position", "delta_to_pole", "made_q3"
    ]],
    on=["season", "round", "DriverNumber"],
    how="left"
)

# -----------------------------
# TARGET & FEATURES
# -----------------------------
TARGET = "Points"

FEATURES = [
    # Driver & team form
    "drv_avg_points_l5",
    "drv_avg_finish_l5",
    "team_avg_points_l5",
    # Qualifying context
    "qualifying_position",
    "delta_to_pole",
    "made_q3",
]

# Drop rows with missing features or target
data = df[FEATURES + [TARGET, "round"]].dropna().copy()

X = data[FEATURES]
y = data[TARGET]

# -----------------------------
# TIME-AWARE VALIDATION
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []
rmse_scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, preds))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))

# -----------------------------
# RESULTS
# -----------------------------
print("📊 QUALIFYING-ENHANCED BASELINE — Points Prediction\n")
print(f"MAE  (avg): {np.mean(mae_scores):.3f}")
print(f"RMSE (avg): {np.mean(rmse_scores):.3f}\n")

print("Features used:")
for f in FEATURES:
    print(f" - {f}")
