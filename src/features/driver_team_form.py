import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
SEASON = 2025
WINDOW = 5

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RACE_FILE = RAW_DIR / f"race_results_{SEASON}.csv"
OUTPUT_FILE = PROCESSED_DIR / f"form_features_{SEASON}.csv"

# -----------------------------
# LOAD
# -----------------------------
race = pd.read_csv(RACE_FILE)

# Ensure correct ordering
race = race.sort_values(["season", "round", "DriverNumber"]).reset_index(drop=True)

# Binary DNF indicator (Status not 'Finished')
race["is_dnf"] = (race["Status"].str.lower() != "finished").astype(int)

# -----------------------------
# DRIVER-LEVEL ROLLING FEATURES
# -----------------------------
def add_driver_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("DriverNumber", group_keys=False)

    df["drv_avg_finish_l5"] = g["finish_position"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )
    df["drv_avg_points_l5"] = g["Points"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )
    df["drv_avg_pos_gained_l5"] = g["positions_gained"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )
    df["drv_finish_std_l5"] = g["finish_position"].apply(
        lambda s: s.shift(1).rolling(window).std()
    )
    df["drv_dnf_rate_l5"] = g["is_dnf"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )

    return df

race = add_driver_form(race, WINDOW)

# -----------------------------
# TEAM-LEVEL ROLLING FEATURES
# -----------------------------
def add_team_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("TeamName", group_keys=False)

    df["team_avg_finish_l5"] = g["finish_position"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )
    df["team_avg_points_l5"] = g["Points"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )
    df["team_avg_pos_gained_l5"] = g["positions_gained"].apply(
        lambda s: s.shift(1).rolling(window).mean()
    )

    return df

race = add_team_form(race, WINDOW)

# -----------------------------
# FINAL SELECT & SAVE
# -----------------------------
cols = [
    "season", "round", "race_name",
    "DriverNumber", "Abbreviation", "FullName", "TeamName",
    "finish_position", "grid_position", "positions_gained", "Points", "Status",
    # Driver form
    "drv_avg_finish_l5", "drv_avg_points_l5",
    "drv_avg_pos_gained_l5", "drv_finish_std_l5", "drv_dnf_rate_l5",
    # Team form
    "team_avg_finish_l5", "team_avg_points_l5", "team_avg_pos_gained_l5",
]

race[cols].to_csv(OUTPUT_FILE, index=False)

print("🎉 PHASE 3 COMPLETE — Driver & Team Form Features")
print(f"📁 Output file : {OUTPUT_FILE}")
print(f"📊 Rows        : {len(race)}")
print(f"🧠 Window size : {WINDOW}")
