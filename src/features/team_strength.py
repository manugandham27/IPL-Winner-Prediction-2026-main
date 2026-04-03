"""
Team batting and bowling strength features derived from player_stats table.

For each team × season, computes:
  - Top-3 batsmen average (higher → better batting depth)
  - Top-3 bowlers economy (lower → better bowling attack)
  - Batting vs bowling balance score

These are causal features: a team with high batting avg will tend to win.
Unlike win-rate features (which are outcomes), these capture WHY a team is strong.
"""
import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SQLITE_DB_PATH, RAW_DIR, EXPECTED_XI_2026

# IPL average batting avg and economy (for normalization)
IPL_AVG_BATTING_AVG = 28.0
IPL_AVG_ECONOMY     = 8.5
IPL_AVG_SR          = 135.0


@lru_cache(maxsize=None)
def load_player_stats_cache() -> pd.DataFrame:
    """Load player stats once and cache."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        "SELECT season, player_name, team, role, batting_avg, batting_sr, "
        "       runs_scored, wickets, bowling_avg, economy "
        "FROM player_stats ORDER BY season, team",
        conn,
    )
    conn.close()
    return df


@lru_cache(maxsize=None)
def load_match_rosters() -> dict:
    path = os.path.join(RAW_DIR, "match_rosters.json")
    if os.path.exists(path):
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

@lru_cache(maxsize=None)
def load_phase_stats() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "player_stats_phases.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def get_team_batting_strength(team: str, season: int, roster: list) -> float:
    df = load_player_stats_cache()
    if roster:
        # Get historical average of these specific players
        players = df[(df["player_name"].isin(roster)) & (df["season"] < season) & (df["batting_avg"] > 0)]
        if len(players) > 0:
            # group by player to get their average over years, then mean of top 5
            avgs = players.groupby("player_name")["batting_avg"].mean().nlargest(5)
            return float(np.clip(avgs.mean() / 60.0, 0, 1))
            
    batsmen = df[(df["team"] == team) & (df["season"] == season - 1) & (df["batting_avg"] > 0)].nlargest(3, "batting_avg")
    if len(batsmen) == 0:
        return IPL_AVG_BATTING_AVG / 60.0
    return float(np.clip(batsmen["batting_avg"].mean() / 60.0, 0, 1))

def get_team_bowling_strength(team: str, season: int, roster: list) -> float:
    df = load_player_stats_cache()
    if roster:
        players = df[(df["player_name"].isin(roster)) & (df["season"] < season) & (df["economy"] > 0)]
        if len(players) > 0:
            econ = players.groupby("player_name")["economy"].mean().nsmallest(5)
            return float(np.clip((12.0 - econ.mean()) / 6.0, 0, 1))
            
    bowlers = df[(df["team"] == team) & (df["season"] == season - 1) & (df["wickets"] > 0) & (df["economy"] > 0)].nlargest(3, "wickets")
    if len(bowlers) == 0:
        return (12.0 - IPL_AVG_ECONOMY) / 6.0
    return float(np.clip((12.0 - bowlers["economy"].mean()) / 6.0, 0, 1))


def get_team_phase_strength(team: str, season: int, roster: list, phase: str, metric: str) -> float:
    """metric can be 'batting' or 'bowling'"""
    phases_df = load_phase_stats()
    if len(phases_df) == 0:
        return 0.5
        
    if roster:
        players = phases_df[(phases_df["player_name"].isin(roster)) & (phases_df["season"] < season) & (phases_df["phase"] == phase)]
    else:
        players = phases_df[(phases_df["season"] < season) & (phases_df["phase"] == phase)] # Approx without roster
        
    if metric == "batting":
        runs = players["runs_scored"].sum()
        balls = players["balls_faced"].sum()
        if balls == 0: return 0.5
        sr = (runs / balls) * 100
        return float(np.clip(sr / 200.0, 0, 1))
    else:
        runs = players["runs_conceded"].sum()
        balls = players["balls_bowled"].sum()
        if balls == 0: return 0.5
        econ = runs / (balls / 6)
        return float(np.clip((15.0 - econ) / 10.0, 0, 1))

def get_team_strength_features(team: str, season: int, match_id: int = None) -> dict:
    rosters = load_match_rosters()
    if match_id and match_id in rosters:
        roster = rosters[match_id]
    else:
        # Fallback to expected XI if it's 2026 or a future prediction without a match ID
        roster = EXPECTED_XI_2026.get(team, [])

    bat = get_team_batting_strength(team, season, roster)
    bowl = get_team_bowling_strength(team, season, roster)
    
    pp_bat = get_team_phase_strength(team, season, roster, "Powerplay", "batting")
    pp_bowl = get_team_phase_strength(team, season, roster, "Powerplay", "bowling")
    death_bat = get_team_phase_strength(team, season, roster, "Death", "batting")
    death_bowl = get_team_phase_strength(team, season, roster, "Death", "bowling")

    return {
        "batting_strength": bat,
        "bowling_strength": bowl,
        "bat_bowl_balance": abs(bat - bowl),
        "pp_batting_str": pp_bat,
        "pp_bowling_str": pp_bowl,
        "death_batting_str": death_bat,
        "death_bowling_str": death_bowl,
    }
