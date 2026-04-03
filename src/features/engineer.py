"""
Feature engineering for IPL match prediction.

Features generated per match row (team1 vs team2):
 - Toss win / decision
 - Win percentage: all-time (smoothed), last 3 seasons, last 5 matches
 - Head-to-head win rate (last 3 seasons)
 - Home ground advantage
 - Recent titles (last 5 seasons only)
 - Season form
 - Venue win rate for each team
 - Venue pitch features
 - Team batting/bowling strength (from real player stats)
"""
import os
import sys
import sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    PROCESSED_MATCHES_CSV, FEATURES_CSV,
    PROCESSED_DIR, FORM_WINDOW, H2H_WINDOW_SEASONS, SQLITE_DB_PATH,
)
from src.features.venue_features import (
    get_venue_avg_score, get_venue_toss_impact, get_venue_size,
    get_recent_venue_avg_score, get_recent_venue_toss_impact
)
from src.features.team_strength import get_team_strength_features

# Kept for backward-compatible tests and reporting utilities.
# This constant is intentionally not used as a predictive feature.
TITLE_COUNTS = {
    "CSK": 5,
    "MI": 5,
    "KKR": 3,
    "SRH": 1,
    "RR": 1,
    "GT": 1,
    "RCB": 0,
    "DC": 0,
    "PBKS": 0,
    "LSG": 0,
}


def calculate_elo_ratings(matches: pd.DataFrame, initial_elo=1500, k_factor=32) -> tuple:
    """Calculate Elo ratings sequentially over matches."""
    elo_dict = {}
    elo_history = {} # match_id -> (t1_elo_before, t2_elo_before)

    for idx, row in matches.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row.get("winner")
        if t1 not in elo_dict: elo_dict[t1] = initial_elo
        if t2 not in elo_dict: elo_dict[t2] = initial_elo

        elo_history[idx] = (elo_dict[t1], elo_dict[t2])

        if pd.isna(winner) or winner == "":
            continue # No update for ties without winner

        # Expected score
        expected_t1 = 1 / (1 + 10 ** ((elo_dict[t2] - elo_dict[t1]) / 400))
        expected_t2 = 1 - expected_t1

        # Actual score
        actual_t1 = 1 if winner == t1 else 0
        actual_t2 = 1 - actual_t1

        # Update
        elo_dict[t1] += k_factor * (actual_t1 - expected_t1)
        elo_dict[t2] += k_factor * (actual_t2 - expected_t2)

    return elo_history, elo_dict


def get_last_n_seasons_wr(matches: pd.DataFrame, team: str,
                           before_season: int, n_seasons: int = 3) -> float:
    """Win rate over the N most recent completed seasons before `before_season`."""
    relevant = matches[(matches["season"] < before_season) &
                       ((matches["team1"] == team) | (matches["team2"] == team))]
    if len(relevant) == 0:
        return 0.5

    available_seasons = sorted(relevant["season"].unique())[-n_seasons:]
    recent = relevant[relevant["season"].isin(available_seasons)]

    if len(recent) == 0:
        return 0.5

    wins = (recent["winner"] == team).sum()
    prior_weight = 4
    return (wins + prior_weight * 0.5) / (len(recent) + prior_weight)


def get_recent_form(matches: pd.DataFrame, team: str, before_idx: int, n: int = 5) -> float:
    """Win rate of `team` in its last `n` matches before row index `before_idx`."""
    past = matches.iloc[:before_idx]
    team_matches = past[(past["team1"] == team) | (past["team2"] == team)]
    recent = team_matches.tail(n)
    if len(recent) == 0:
        return 0.5
    wins = (recent["winner"] == team).sum()
    return wins / len(recent)


def get_h2h_rate(matches: pd.DataFrame, team1: str, team2: str, before_idx: int,
                 window_seasons: int = 3) -> float:
    """Head-to-head win rate of team1 vs team2 over last `window_seasons` seasons."""
    past = matches.iloc[:before_idx]
    if len(past) == 0:
        return 0.5
    recent_seasons = sorted(past["season"].unique())[-window_seasons:]
    h2h = past[
        (past["season"].isin(recent_seasons)) &
        (((past["team1"] == team1) & (past["team2"] == team2)) |
         ((past["team1"] == team2) & (past["team2"] == team1)))
    ]
    if len(h2h) == 0:
        return 0.5
    wins1 = (h2h["winner"] == team1).sum()
    return wins1 / len(h2h)


def get_venue_win_rate(matches: pd.DataFrame, team: str, venue: str, before_idx: int) -> float:
    """Win rate of `team` at a specific venue."""
    past = matches.iloc[:before_idx]
    at_venue = past[
        (past["venue"] == venue) &
        ((past["team1"] == team) | (past["team2"] == team))
    ]
    if len(at_venue) == 0:
        return 0.5
    wins = (at_venue["winner"] == team).sum()
    return wins / len(at_venue)


def is_home_ground(team: str, venue: str) -> int:
    """Returns 1 if venue is team's home ground."""
    home_map = {
        "CSK":  ["MA Chidambaram Stadium"],
        "MI":   ["Wankhede Stadium"],
        "RCB":  ["M Chinnaswamy Stadium"],
        "KKR":  ["Eden Gardens"],
        "DC":   ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
        "RR":   ["Sawai Mansingh Stadium"],
        "SRH":  ["Rajiv Gandhi International Cricket Stadium"],
        "PBKS": ["Punjab Cricket Association IS Bindra Stadium",
                 "Punjab Cricket Association Stadium"],
        "GT":   ["Narendra Modi Stadium"],
        "LSG":  ["BRSABV Ekana Cricket Stadium"],
    }
    return int(venue in home_map.get(team, []))


def get_season_form(matches: pd.DataFrame, team: str, season: int, before_idx: int) -> float:
    """Win rate within the current season up to before_idx."""
    season_matches = matches.iloc[:before_idx]
    season_matches = season_matches[season_matches["season"] == season]
    team_matches = season_matches[(season_matches["team1"] == team) | (season_matches["team2"] == team)]
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return wins / len(team_matches)


def load_champions_by_season() -> dict:
    """Load season champions from SQLite season_stats table."""
    if not os.path.exists(SQLITE_DB_PATH):
        return {}
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT season, team FROM season_stats WHERE won_title = 1"
        ).fetchall()
        return {int(season): team for season, team in rows}
    finally:
        conn.close()


def get_recent_titles(team: str, before_season: int, champions_by_season: dict = None,
                      window: int = 5) -> int:
    """
    Titles won by this team in the `window` seasons before `before_season`.
    If champions_by_season not provided, loads from DB.
    """
    if champions_by_season is None:
        champions_by_season = load_champions_by_season()
    count = 0
    for season in range(before_season - window, before_season):
        if champions_by_season.get(season) == team:
            count += 1
    return count


def build_features(matches_csv: str = PROCESSED_MATCHES_CSV) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)
    df = df.reset_index(drop=True)
    champions_by_season = load_champions_by_season()

    elo_history, current_elo = calculate_elo_ratings(df)

    rows = []
    for idx, row in df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        venue = row.get("venue", "")
        season = int(row["season"])

        f = {}

        f["match_id"] = row["match_id"]
        f["season"] = season
        f["team1"] = t1
        f["team2"] = t2

        # Toss features
        f["toss_won_by_team1"] = int(row.get("toss_won_by_team1", 0))
        f["toss_decision_bat"] = int(row.get("toss_decision_bat", 0))

        # Elo feature
        t1_elo, t2_elo = elo_history[idx]
        f["t1_elo"] = t1_elo
        f["t2_elo"] = t2_elo
        f["elo_diff"] = t1_elo - t2_elo

        # Last 3 seasons win rate
        f["t1_last3yr_wr"] = get_last_n_seasons_wr(df, t1, season, n_seasons=3)
        f["t2_last3yr_wr"] = get_last_n_seasons_wr(df, t2, season, n_seasons=3)
        f["last3yr_wr_diff"] = f["t1_last3yr_wr"] - f["t2_last3yr_wr"]

        # Recent match form (last 5 matches)
        f["t1_recent_form"] = get_recent_form(df, t1, idx, FORM_WINDOW)
        f["t2_recent_form"] = get_recent_form(df, t2, idx, FORM_WINDOW)
        f["form_diff"] = f["t1_recent_form"] - f["t2_recent_form"]

        # Season form
        f["t1_season_form"] = get_season_form(df, t1, season, idx)
        f["t2_season_form"] = get_season_form(df, t2, season, idx)

        # Head-to-head
        f["h2h_t1_wr"] = get_h2h_rate(df, t1, t2, idx, H2H_WINDOW_SEASONS)

        # Venue win rates
        f["t1_venue_wr"] = get_venue_win_rate(df, t1, venue, idx)
        f["t2_venue_wr"] = get_venue_win_rate(df, t2, venue, idx)
        f["venue_wr_diff"] = f["t1_venue_wr"] - f["t2_venue_wr"]

        # Home advantage
        f["t1_is_home"] = is_home_ground(t1, venue)
        f["t2_is_home"] = is_home_ground(t2, venue)

        # Recent titles (last 5 seasons)
        f["t1_recent_titles"]  = get_recent_titles(t1, season, champions_by_season, window=5)
        f["t2_recent_titles"]  = get_recent_titles(t2, season, champions_by_season, window=5)
        f["recent_title_diff"] = f["t1_recent_titles"] - f["t2_recent_titles"]

        # Venue pitch features (Updated to Recent Venue Form)
        f["venue_avg_score"] = get_recent_venue_avg_score(df, venue, idx)
        f["venue_toss_impact"] = get_recent_venue_toss_impact(df, venue, idx)
        f["venue_size"] = get_venue_size(venue)

        # Team batting/bowling strength and phase variables
        t1_str = get_team_strength_features(t1, season, row["match_id"])
        t2_str = get_team_strength_features(t2, season, row["match_id"])
        
        f["t1_batting_str"] = t1_str["batting_strength"]
        f["t2_batting_str"] = t2_str["batting_strength"]
        f["batting_str_diff"] = t1_str["batting_strength"] - t2_str["batting_strength"]
        f["t1_bowling_str"] = t1_str["bowling_strength"]
        f["t2_bowling_str"] = t2_str["bowling_strength"]
        f["bowling_str_diff"] = t1_str["bowling_strength"] - t2_str["bowling_strength"]
        
        # New Phase variables
        f["t1_pp_batting"] = t1_str["pp_batting_str"]
        f["t2_pp_batting"] = t2_str["pp_batting_str"]
        f["pp_batting_diff"] = t1_str["pp_batting_str"] - t2_str["pp_batting_str"]
        
        f["t1_death_bowling"] = t1_str["death_bowling_str"]
        f["t2_death_bowling"] = t2_str["death_bowling_str"]
        f["death_bowling_diff"] = t1_str["death_bowling_str"] - t2_str["death_bowling_str"]

        # Target
        f["team1_won"] = int(row["team1_won"])

        rows.append(f)

    features_df = pd.DataFrame(rows)
    return features_df


def save_features(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Features saved: {len(df)} rows x {len(df.columns)} cols -> {FEATURES_CSV}")
    feature_cols = [c for c in df.columns
                    if c not in ("match_id", "season", "team1", "team2", "team1_won")]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


def run_feature_engineering() -> pd.DataFrame:
    df = build_features()
    save_features(df)
    return df


if __name__ == "__main__":
    run_feature_engineering()
