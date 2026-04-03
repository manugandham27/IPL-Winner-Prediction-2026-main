"""
Single match predictor: given two team names, predicts the winner
using the trained ensemble model and shows detailed team stats.

Usage:
  python main.py --mode match --team1 CSK --team2 PBKS

  from src.prediction.match_predictor import predict_match
  result = predict_match("CSK", "PBKS")
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROCESSED_MATCHES_CSV, FEATURES_CSV, TEAMS, EXPECTED_XI_2026
from src.models.base_model import FEATURE_COLS
from src.prediction.predict_2026 import build_matchup_features
from src.features.engineer import calculate_elo_ratings, get_last_n_seasons_wr, get_recent_form
from src.features.team_strength import get_team_strength_features


def get_h2h_record(df, team1, team2):
    """Get head-to-head record between two teams."""
    mask = ((df["team1"] == team1) & (df["team2"] == team2)) | \
           ((df["team1"] == team2) & (df["team2"] == team1))
    h2h = df[mask]
    t1_wins = (h2h["winner"] == team1).sum()
    t2_wins = (h2h["winner"] == team2).sum()
    total = len(h2h)
    return t1_wins, t2_wins, total


def get_last_n_matches(df, team, n=5):
    """Get results of the last N matches for a team."""
    mask = (df["team1"] == team) | (df["team2"] == team)
    team_matches = df[mask].tail(n)
    results = []
    for _, row in team_matches.iterrows():
        won = row["winner"] == team
        opponent = row["team2"] if row["team1"] == team else row["team1"]
        results.append({"opponent": opponent, "won": won})
    return results


def predict_match(team1: str, team2: str, venue: str = None,
                  toss_winner: str = None, toss_decision: str = "field") -> dict:
    """
    Predict the winner of a single match with detailed stats.

    Args:
        team1: Team abbreviation (e.g. 'CSK')
        team2: Team abbreviation (e.g. 'PBKS')
        venue: Optional venue name
        toss_winner: Optional toss winner
        toss_decision: 'bat' or 'field' (default: 'field')

    Returns:
        dict with probabilities, stats, and predicted winner
    """
    from src.models.ensemble_model import EnsembleModel

    try:
        model = EnsembleModel()
        model.load()
    except FileNotFoundError:
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel()
        model.load()

    matches_df = pd.read_csv(PROCESSED_MATCHES_CSV)

    # Build features
    feats = build_matchup_features(team1, team2, matches_df)

    # Override toss if specified
    if toss_winner is not None:
        feats["toss_won_by_team1"] = int(toss_winner == team1)
    if toss_decision:
        feats["toss_decision_bat"] = int(toss_decision == "bat")

    # Predict
    probs = model.predict_proba(feats)
    t1_prob = probs[:, 1].mean()
    t2_prob = 1 - t1_prob
    winner = team1 if t1_prob >= 0.5 else team2

    # --- Gather detailed stats ---
    _, current_elo = calculate_elo_ratings(matches_df)
    t1_elo = current_elo.get(team1, 1500)
    t2_elo = current_elo.get(team2, 1500)

    t1_last3yr = get_last_n_seasons_wr(matches_df, team1, 2026, 3)
    t2_last3yr = get_last_n_seasons_wr(matches_df, team2, 2026, 3)

    t1_form = get_recent_form(matches_df, team1, len(matches_df), 5)
    t2_form = get_recent_form(matches_df, team2, len(matches_df), 5)

    h2h_t1, h2h_t2, h2h_total = get_h2h_record(matches_df, team1, team2)

    t1_str = get_team_strength_features(team1, 2025)
    t2_str = get_team_strength_features(team2, 2025)

    t1_recent = get_last_n_matches(matches_df, team1, 5)
    t2_recent = get_last_n_matches(matches_df, team2, 5)

    t1_xi = EXPECTED_XI_2026.get(team1, [])
    t2_xi = EXPECTED_XI_2026.get(team2, [])

    return {
        "team1": team1,
        "team1_name": TEAMS.get(team1, team1),
        "team1_win_prob": round(t1_prob * 100, 2),
        "team2": team2,
        "team2_name": TEAMS.get(team2, team2),
        "team2_win_prob": round(t2_prob * 100, 2),
        "predicted_winner": winner,
        "predicted_winner_name": TEAMS.get(winner, winner),
        "confidence": round(max(t1_prob, t2_prob) * 100, 2),
        # Stats
        "t1_elo": round(t1_elo, 1),
        "t2_elo": round(t2_elo, 1),
        "t1_last3yr_wr": round(t1_last3yr * 100, 1),
        "t2_last3yr_wr": round(t2_last3yr * 100, 1),
        "t1_recent_form": round(t1_form * 100, 1),
        "t2_recent_form": round(t2_form * 100, 1),
        "h2h_t1_wins": h2h_t1,
        "h2h_t2_wins": h2h_t2,
        "h2h_total": h2h_total,
        "t1_batting_str": round(t1_str["batting_strength"] * 100, 1),
        "t2_batting_str": round(t2_str["batting_strength"] * 100, 1),
        "t1_bowling_str": round(t1_str["bowling_strength"] * 100, 1),
        "t2_bowling_str": round(t2_str["bowling_strength"] * 100, 1),
        "t1_pp_batting": round(t1_str["pp_batting_str"] * 100, 1),
        "t2_pp_batting": round(t2_str["pp_batting_str"] * 100, 1),
        "t1_death_bowling": round(t1_str["death_bowling_str"] * 100, 1),
        "t2_death_bowling": round(t2_str["death_bowling_str"] * 100, 1),
        "t1_recent_matches": t1_recent,
        "t2_recent_matches": t2_recent,
        "t1_xi": t1_xi,
        "t2_xi": t2_xi,
    }


def print_match_result(result: dict):
    t1 = result["team1"]
    t2 = result["team2"]
    t1n = result["team1_name"]
    t2n = result["team2_name"]

    print()
    print("=" * 65)
    print(f"  🏏  MATCH PREDICTION: {t1} vs {t2}")
    print("=" * 65)

    # Win probability bar
    t1p = result["team1_win_prob"]
    t2p = result["team2_win_prob"]
    bar_len = 40
    t1_bar = int(t1p / 100 * bar_len)
    t2_bar = bar_len - t1_bar
    print(f"\n  Win Probability:")
    print(f"  {t1n:<25} {t1p:>6.2f}%  {'█' * t1_bar}{'░' * t2_bar}")
    print(f"  {t2n:<25} {t2p:>6.2f}%  {'░' * t1_bar}{'█' * t2_bar}")

    # Head to Head
    print(f"\n  ── Head-to-Head Record ({result['h2h_total']} matches) ──")
    print(f"  {t1}: {result['h2h_t1_wins']} wins    {t2}: {result['h2h_t2_wins']} wins")

    # Key stats comparison table
    print(f"\n  ── Key Stats Comparison ──")
    print(f"  {'Metric':<25} {t1:>8}  {t2:>8}")
    print(f"  {'─' * 45}")
    print(f"  {'Elo Rating':<25} {result['t1_elo']:>8.1f}  {result['t2_elo']:>8.1f}")
    print(f"  {'Last 3yr Win Rate':<25} {result['t1_last3yr_wr']:>7.1f}%  {result['t2_last3yr_wr']:>7.1f}%")
    print(f"  {'Recent Form (Last 5)':<25} {result['t1_recent_form']:>7.1f}%  {result['t2_recent_form']:>7.1f}%")
    print(f"  {'Batting Strength':<25} {result['t1_batting_str']:>7.1f}%  {result['t2_batting_str']:>7.1f}%")
    print(f"  {'Bowling Strength':<25} {result['t1_bowling_str']:>7.1f}%  {result['t2_bowling_str']:>7.1f}%")
    print(f"  {'PowerPlay Batting':<25} {result['t1_pp_batting']:>7.1f}%  {result['t2_pp_batting']:>7.1f}%")
    print(f"  {'Death Bowling':<25} {result['t1_death_bowling']:>7.1f}%  {result['t2_death_bowling']:>7.1f}%")

    # Recent form streak
    print(f"\n  ── Recent Form ──")
    for label, key in [(t1, "t1_recent_matches"), (t2, "t2_recent_matches")]:
        matches = result[key]
        streak = ""
        for m in matches:
            streak += "✅" if m["won"] else "❌"
        opponents = ", ".join([f"{'W' if m['won'] else 'L'} vs {m['opponent']}" for m in matches])
        print(f"  {label}: {streak}  ({opponents})")

    # Expected XI
    for label, key in [(t1n, "t1_xi"), (t2n, "t2_xi")]:
        xi = result[key]
        if xi:
            print(f"\n  ── Expected XI: {label} ──")
            for i, player in enumerate(xi, 1):
                print(f"  {i:>2}. {player}")

    # Final verdict
    winner = result["predicted_winner_name"]
    conf = result["confidence"]
    print()
    print("=" * 65)
    print(f"  🏆  PREDICTED WINNER: {winner}  ({conf:.2f}% confidence)")
    print("=" * 65)
    print()


if __name__ == "__main__":
    import sys
    t1 = sys.argv[1] if len(sys.argv) > 1 else "CSK"
    t2 = sys.argv[2] if len(sys.argv) > 2 else "PBKS"
    result = predict_match(t1, t2)
    print_match_result(result)
