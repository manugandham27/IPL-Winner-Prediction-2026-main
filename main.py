"""
IPL 2026 Winner Prediction - Main Entry Point

Uses REAL IPL ball-by-ball data (IPL.csv, 2008-2025, 1169 matches).

Usage:
  python main.py --mode setup      # Extract data from IPL.csv and engineer features
  python main.py --mode train      # Train all models
  python main.py --mode predict    # Predict 2026 winner
  python main.py --mode match --team1 CSK --team2 PBKS   # Predict a specific match
  python main.py --mode all        # Run full pipeline end-to-end
  python main.py --mode visualize  # Generate charts (needs predict to run first)
"""
import argparse
import logging
import os
import sys
import time

from config import LOG_FILE, LOG_LEVEL

# Logging setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def mode_setup():
    logger.info("=== SETUP: Extracting real IPL data and building features ===")
    t0 = time.time()

    from src.data.create_dataset import (
        save_teams_json, build_all_matches, save_matches_csv, save_player_stats_csv,
        save_match_rosters, save_player_stats_phases_csv
    )
    from src.data.db_setup     import setup_database
    from src.data.ingest       import run_ingestion
    from src.data.preprocess   import run_preprocessing
    from src.features.engineer import run_feature_engineering

    logger.info("Step 1/5: Extracting match data from IPL.csv...")
    save_teams_json()
    matches, player_stats, rosters, phase_stats = build_all_matches(return_format="dataframes")
    save_matches_csv(matches)
    save_player_stats_csv(player_stats)
    save_match_rosters(rosters)
    save_player_stats_phases_csv(phase_stats)

    logger.info("Step 2/5: Creating SQLite database schema...")
    setup_database()

    logger.info("Step 3/5: Ingesting data into SQLite...")
    run_ingestion()

    logger.info("Step 4/5: Preprocessing matches...")
    run_preprocessing()

    logger.info("Step 5/5: Engineering features...")
    run_feature_engineering()

    logger.info(f"Setup complete in {time.time()-t0:.1f}s")


def mode_train():
    logger.info("=== TRAIN: Training all models ===")
    t0 = time.time()

    from src.models.trainer import run_training
    results = run_training()

    logger.info(f"Training complete in {time.time()-t0:.1f}s")
    return results


def mode_predict():
    logger.info("=== PREDICT: IPL 2026 Winner Prediction ===")
    t0 = time.time()

    from src.prediction.predict_2026 import (
        predict_2026_winner, print_predictions, save_predictions,
    )
    rankings = predict_2026_winner()
    print_predictions(rankings)
    save_predictions(rankings)

    logger.info(f"Prediction complete in {time.time()-t0:.1f}s")
    return rankings


def mode_visualize():
    logger.info("=== VISUALIZE: Generating charts ===")
    from src.prediction.visualize import generate_all_charts
    generate_all_charts()


def mode_match(team1: str, team2: str):
    logger.info(f"=== MATCH PREDICTION: {team1} vs {team2} ===")
    from src.prediction.match_predictor import predict_match, print_match_result
    result = predict_match(team1, team2)
    print_match_result(result)
    return result


def mode_all():
    mode_setup()
    mode_train()
    rankings = mode_predict()
    try:
        mode_visualize()
    except Exception as e:
        logger.warning(f"Visualization failed (non-critical): {e}")
    return rankings


def parse_args():
    parser = argparse.ArgumentParser(
        description="IPL 2026 Winner Prediction (Real Data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "train", "predict", "match", "visualize", "all"],
        default="all",
        help="Pipeline mode to run (default: all)",
    )
    parser.add_argument(
        "--team1",
        type=str,
        default=None,
        help="Team 1 abbreviation for match mode (e.g. CSK, MI, RCB, KKR, DC, PBKS, RR, SRH, LSG, GT)",
    )
    parser.add_argument(
        "--team2",
        type=str,
        default=None,
        help="Team 2 abbreviation for match mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting IPL 2026 prediction pipeline | mode={args.mode}")

    if args.mode == "setup":
        mode_setup()
    elif args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict()
    elif args.mode == "match":
        if not args.team1 or not args.team2:
            print("\nError: --team1 and --team2 are required for match mode.")
            print("Example: python main.py --mode match --team1 CSK --team2 PBKS")
            print("\nAvailable teams: CSK, MI, RCB, KKR, DC, PBKS, RR, SRH, LSG, GT")
            sys.exit(1)
        mode_match(args.team1.upper(), args.team2.upper())
    elif args.mode == "visualize":
        mode_visualize()
    elif args.mode == "all":
        mode_all()
