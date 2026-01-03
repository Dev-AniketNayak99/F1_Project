import fastf1
import pandas as pd
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# CONFIG
# -----------------------------
SEASON = 2025
SESSION_TYPE = "Q"
MAX_WORKERS = 8

CACHE_DIR = Path("data/raw/cache")
OUTPUT_DIR = Path("data/raw")

# -----------------------------
# LOGGING (silence FastF1 spam)
# -----------------------------
for logger_name in [
    "fastf1", "fastf1.core", "fastf1.api",
    "fastf1.req", "fastf1._api", "fastf1.events"
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# -----------------------------
# SETUP
# -----------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def _to_seconds(td):
    return td.total_seconds() if pd.notna(td) else None


def fetch_single_qualifying(season: int, event):
    """Fetch official qualifying results for one race."""
    event_name = event["EventName"]
    round_number = event["RoundNumber"]

    try:
        session = fastf1.get_session(season, event_name, SESSION_TYPE)
        session.load()

        results = session.results
        if results is None or results.empty:
            return None

        df = results[[
            "DriverNumber",
            "Abbreviation",
            "FullName",
            "TeamName",
            "Position",
            "Q1",
            "Q2",
            "Q3",
        ]].copy()

        df.rename(columns={"Position": "qualifying_position"}, inplace=True)

        df["q1_time"] = df["Q1"].apply(_to_seconds)
        df["q2_time"] = df["Q2"].apply(_to_seconds)
        df["q3_time"] = df["Q3"].apply(_to_seconds)
        df.drop(columns=["Q1", "Q2", "Q3"], inplace=True)

        df["made_q3"] = df["q3_time"].notna().astype(int)

        pole_time = df[["q1_time", "q2_time", "q3_time"]].min(axis=1).min()
        df["delta_to_pole"] = (
            df[["q1_time", "q2_time", "q3_time"]].min(axis=1) - pole_time
        )

        df["season"] = season
        df["round"] = round_number
        df["race_name"] = event_name

        return df

    except Exception:
        return None


def fetch_qualifying_results(season: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(season)

    # ✅ Filter ONLY race weekends
    schedule = schedule[schedule["EventFormat"] != "testing"]

    print(f"\n🏁 Fetching F1 {season} qualifying results")
    print(f"📅 Race events: {len(schedule)}")
    print(f"⚡ Thread workers: {MAX_WORKERS}\n")

    frames = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_single_qualifying, season, event)
            for _, event in schedule.iterrows()
        ]

        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                frames.append(df)
                print(f"✅ Loaded: {df['race_name'].iloc[0]}")

    if not frames:
        print("⚠️ No qualifying data available yet for this season.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_qualifying_results(SEASON)

    output_path = OUTPUT_DIR / f"qualifying_results_{SEASON}.csv"
    df.to_csv(output_path, index=False)

    print("\n🎉 QUALIFYING INGESTION COMPLETE")
    print(f"📁 Output file : {output_path}")
    print(f"📊 Total rows : {len(df)}")
