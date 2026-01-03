import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

# Drop rows with missing values
data = df[FEATURES + [TARGET]].dropna().copy()

X = data[FEATURES]
y = data[TARGET]

# -----------------------------
# STANDARDIZE FEATURES
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# FIT LINEAR MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_scaled, y)

# -----------------------------
# COEFFICIENT TABLE
# -----------------------------
coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

print("\n📊 BASELINE LINEAR MODEL — FEATURE INTERPRETATION\n")
print(coef_df[["feature", "coefficient"]].to_string(index=False))

print("\nℹ️ Interpretation notes:")
print("- Positive coefficient → increases expected points")
print("- Negative coefficient → decreases expected points")
print("- Coefficients are standardized (comparable magnitudes)")
