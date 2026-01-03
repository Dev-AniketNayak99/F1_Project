import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
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

data = df[FEATURES + [TARGET]].dropna().copy()

X = data[FEATURES]
y = data[TARGET]

# -----------------------------
# TRAIN FINAL GBM ON ALL DATA
# -----------------------------
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n📊 GBM FEATURE IMPORTANCE\n")
print(importance_df.to_string(index=False))

# -----------------------------
# PLOT (SINGLE, CLEAN)
# -----------------------------
plt.figure(figsize=(8, 5))
plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)
plt.gca().invert_yaxis()
plt.xlabel("Relative Importance")
plt.title("Gradient Boosting Feature Importance")
plt.tight_layout()
plt.show()
