import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# CONFIG
# -----------------------------
SEASON = 2025

FORM_FILE = Path("data/processed/form_features_2025.csv")
QUALI_FILE = Path("data/raw/qualifying_results_2025.csv")

TARGET = "Points"

FEATURES = [
    "drv_avg_points_l5",
    "drv_avg_finish_l5",
    "team_avg_points_l5",
    "qualifying_position",
    "delta_to_pole",
    "made_q3",
]

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

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, preds))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))

# -----------------------------
# RESULTS
# -----------------------------
print("🚀 GRADIENT BOOSTING MODEL — Points Prediction\n")
print(f"MAE  (avg): {np.mean(mae_scores):.3f}")
print(f"RMSE (avg): {np.mean(rmse_scores):.3f}\n")

print("Model configuration:")
print(" - n_estimators = 200")
print(" - learning_rate = 0.05")
print(" - max_depth = 3")
print(" - subsample = 0.8")
