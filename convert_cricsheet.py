"""
Convert Cricsheet CSV2 format into the single IPL.csv expected by this project.

Reads individual match CSVs + _info.csv files from /tmp/ipl_cricsheet/
and produces a consolidated IPL.csv with the columns expected by create_dataset.py:
  match_id, season, year, date, venue, city, innings, ball,
  batting_team, bowling_team, batter, non_striker, bowler,
  runs_batter, balls_faced, valid_ball, runs_bowler, bowler_wicket,
  player_out, match_won_by, win_outcome, toss_winner, toss_decision,
  result_type, stage, over
"""
import os
import glob
import re
import pandas as pd
import numpy as np

SRC_DIR = "/tmp/ipl_cricsheet"
OUT_PATH = os.path.join(os.path.dirname(__file__), "IPL.csv")


def parse_info(info_path):
    """Parse a cricsheet *_info.csv to extract match metadata."""
    meta = {}
    with open(info_path) as f:
        for line in f:
            parts = line.strip().split(",", 2)
            if len(parts) < 3 or parts[0] != "info":
                continue
            key, val = parts[1], parts[2].strip('"')
            if key == "season":
                meta["season"] = val
            elif key == "date":
                meta["date"] = val.replace("/", "-")
            elif key == "venue":
                meta["venue"] = val
            elif key == "city":
                meta["city"] = val
            elif key == "toss_winner":
                meta["toss_winner"] = val
            elif key == "toss_decision":
                meta["toss_decision"] = val
            elif key == "winner":
                meta["match_won_by"] = val
            elif key == "winner_runs":
                meta["win_outcome"] = f"{val} runs"
            elif key == "winner_wickets":
                meta["win_outcome"] = f"{val} wickets"
            elif key == "outcome" and val == "no result":
                meta["result_type"] = "no result"
            elif key == "match_number":
                meta["stage"] = val
    # Fill defaults
    meta.setdefault("win_outcome", "")
    meta.setdefault("match_won_by", "Unknown")
    meta.setdefault("result_type", None)
    meta.setdefault("city", "")
    meta.setdefault("stage", "")
    return meta


def convert():
    # Find all delivery CSVs (not _info ones)
    delivery_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.csv")))
    delivery_files = [f for f in delivery_files if "_info" not in f]

    print(f"Found {len(delivery_files)} match files")

    all_rows = []
    for i, dpath in enumerate(delivery_files):
        match_id_str = os.path.basename(dpath).replace(".csv", "")
        info_path = os.path.join(SRC_DIR, f"{match_id_str}_info.csv")

        if not os.path.exists(info_path):
            continue

        meta = parse_info(info_path)

        try:
            ddf = pd.read_csv(dpath)
        except Exception:
            continue

        if len(ddf) == 0:
            continue

        # Derive year from date
        date_str = meta.get("date", "")
        year = int(date_str[:4]) if len(date_str) >= 4 else 2020

        # Map columns to expected names
        ddf = ddf.rename(columns={
            "striker": "batter",
            "runs_off_bat": "runs_batter",
        })

        # Compute over from ball (ball is like 0.1, 0.2, ..., 19.6)
        if "ball" in ddf.columns:
            ddf["over"] = np.floor(pd.to_numeric(ddf["ball"], errors="coerce")).astype(int)
        else:
            ddf["over"] = 0

        # balls_faced = 1 for each legal delivery
        ddf["valid_ball"] = ((ddf.get("wides", pd.Series(dtype=float)).fillna(0) == 0)).astype(int)
        ddf["balls_faced"] = ddf["valid_ball"]

        # runs_bowler = runs_off_bat + extras - byes - legbyes (runs charged to bowler)
        extras = ddf.get("extras", pd.Series(0, index=ddf.index)).fillna(0)
        byes = ddf.get("byes", pd.Series(0, index=ddf.index)).fillna(0)
        legbyes = ddf.get("legbyes", pd.Series(0, index=ddf.index)).fillna(0)
        ddf["runs_bowler"] = ddf["runs_batter"].fillna(0) + extras - byes - legbyes

        # bowler_wicket
        wt = ddf.get("wicket_type", pd.Series("", index=ddf.index)).fillna("")
        ddf["bowler_wicket"] = wt.apply(
            lambda x: x if x not in ("", "run out", "retired hurt", "retired out", "obstructing the field") else ""
        )

        # player_out
        ddf["player_out"] = ddf.get("player_dismissed", pd.Series("", index=ddf.index)).fillna("")

        # Add metadata columns
        ddf["match_id"] = int(match_id_str)
        ddf["season"] = meta.get("season", str(year))
        ddf["year"] = year
        ddf["date"] = meta.get("date", "")
        ddf["venue"] = meta.get("venue", "")
        ddf["city"] = meta.get("city", "")
        ddf["match_won_by"] = meta.get("match_won_by", "Unknown")
        ddf["win_outcome"] = meta.get("win_outcome", "")
        ddf["toss_winner"] = meta.get("toss_winner", "")
        ddf["toss_decision"] = meta.get("toss_decision", "")
        ddf["result_type"] = meta.get("result_type")
        ddf["stage"] = meta.get("stage", "")

        cols = [
            "match_id", "season", "year", "date", "venue", "city",
            "innings", "ball", "over",
            "batting_team", "bowling_team", "batter", "non_striker", "bowler",
            "runs_batter", "balls_faced", "valid_ball", "runs_bowler",
            "bowler_wicket", "player_out",
            "match_won_by", "win_outcome", "toss_winner", "toss_decision",
            "result_type", "stage",
        ]
        # Only keep columns that exist
        cols = [c for c in cols if c in ddf.columns]
        all_rows.append(ddf[cols])

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(delivery_files)} matches...")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nDone! Wrote {len(combined)} deliveries across {combined['match_id'].nunique()} matches -> {OUT_PATH}")
    print(f"File size: {os.path.getsize(OUT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    convert()
