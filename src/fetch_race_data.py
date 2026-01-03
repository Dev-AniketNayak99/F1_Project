import fastf1
import pandas as pd
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# CONFIG
# -----------------------------
SEASON = 2025
SESSION_TYPE = "R"          # Race
MAX_WORKERS = 6

CACHE_DIR = Path("data/raw/cache")
OUTPUT_DIR = Path("data/raw")

# -----------------------------
# LOGGING (silence FastF1 noise)
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


def fetch_single_race(season: int, event):
    """
    Fetch official race results for a single event.
    Returns DataFrame or None.
    """
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
            "GridPosition",
            "Points",
            "Status",
        ]].copy()

        df.rename(columns={
            "Position": "finish_position",
            "GridPosition": "grid_position"
        }, inplace=True)

        # Derived race metrics (simple but powerful)
        df["positions_gained"] = (
            df["grid_position"] - df["finish_position"]
        )

        df["season"] = season
        df["round"] = round_number
        df["race_name"] = event_name

        return df

    except Exception:
        return None


def fetch_race_results(season: int) -> pd.DataFrame:
    """
    Fetch race results for all race weekends in parallel.
    Handles partial seasons gracefully.
    """
    schedule = fastf1.get_event_schedule(season)

    # ✅ Filter ONLY race weekends
    schedule = schedule[schedule["EventFormat"] != "testing"]

    print(f"\n🏎️ Fetching F1 {season} race results")
    print(f"📅 Race events: {len(schedule)}")
    print(f"⚡ Thread workers: {MAX_WORKERS}\n")

    frames = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_single_race, season, event)
            for _, event in schedule.iterrows()
        ]

        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                frames.append(df)
                print(f"✅ Loaded: {df['race_name'].iloc[0]}")

    if not frames:
        print("⚠️ No race data available yet for this season.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_race_results(SEASON)

    output_path = OUTPUT_DIR / f"race_results_{SEASON}.csv"
    df.to_csv(output_path, index=False)

    print("\n🎉 RACE RESULTS INGESTION COMPLETE")
    print(f"📁 Output file : {output_path}")
    print(f"📊 Total rows : {len(df)}")
