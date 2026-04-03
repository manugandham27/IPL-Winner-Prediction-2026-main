# 🏏 IPL 2026 Winner Prediction & Match Simulator

A comprehensive, production-ready machine learning pipeline designed to predict IPL 2026 outcomes and simulate direct team matchups. 

This pipeline goes far beyond basic seasonal averages by implementing **dynamic player-level rosters**, measuring statistical momentum via **chess-style Elo ratings**, and extracting **Phase-Specific Matchup Data** (PowerPlay vs Death overs) from over 270,000 real ball-by-ball IPL deliveries.

---

## 🌟 Advanced Features Implemented

*   **Dynamic Playing XIs:** The model evaluates team strength based on the **specific 11-12 players** taking the field using `match_rosters` extracting from historical deliveries, rather than using generic, static franchise-level averages.
*   **Chess-Style Elo Ratings:** Replaced naive all-time win rates with a sequential **K-Factor Elo Rating engine**. It tracks true, active team capability by exchanging rating points based on match expectations vs actual outcomes.
*   **Phase-Specific Micro-Matchups:** Performance isn't judged globally. Our pipeline evaluates **Powerplay Batting vs Powerplay Bowling** and **Death Overs execution** dynamically to pinpoint where games are won and lost.
*   **Rolling Venue Conditions:** Pitch evaluation ignores 15-year old data, instead calculating rolling first-innings averages based strictly on the **last 5-10 matches** at any specific stadium to mirror current pitch telemetry.
*   **Live Match Simulator (`--mode match`):** Interactively pit any two IPL franchises together to generate detailed probabilities based on real-time Elo, Head-to-Head records, Venue Form, and simulated Expected XIs!

---

## 🔮 Prediction Modes

The project supports three primary use cases:
1. **Tournament Winner Probabilities:** Ranks all 10 franchises for the 2026 trophy (combining Squad Strength, Form, and Model Projections).
2. **Match-by-Match Fixture Run:** Predicts the outcome of the entire 14-game league-stage timetable (`ipl-2026-UTC.csv`).
3. **Interactive Match Simulator:** Pass two teams and instantly generate deep statistical head-to-head metrics.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Data
*(Note: To respect GitHub's storage constraints, raw multi-megabyte Cricsheet `.csv` delivery files should be placed in `/tmp/ipl_cricsheet` or extracted natively via the `convert_cricsheet.py` data ingestion script to generate the 50MB+ `IPL.csv`.)*
```bash
python convert_cricsheet.py
```

### 3. Run the ML Pipeline End-to-End
This single command cleans data, calculates Elo ratings, extracts Phase-stats, trains 5 distinct ML models + a Stacking Ensemble, predicts the 2026 winner, and renders `matplotlib` visuals:
```bash
python main.py --mode all
```

*You can also run independent stages:*
```bash
python main.py --mode setup      # Extract data, parse rosters, build database
python main.py --mode train      # Train Random Forest, XGBoost, LightGBM, ExtraTrees, NN
python main.py --mode predict    # Predict IPL 2026 winner probabilities
```

### 4. Interactive Match Simulator (New!)
Instantly predict tonight's matchup and get a beautiful terminal printout comparing Elo, Phase Strengths, Head-to-Head and the expected 2026 Playing XIs:
```bash
python main.py --mode match --team1 CSK --team2 PBKS
```

---

## 📊 Model Architecture & Performance
Stacking Ensemble utilizing LightGBM, XGBoost, Scikit-Learn pipelines, and Neural Networks trained on **1,175 IPL Matches**:

| Model | CV Accuracy | Test Accuracy | Test AUC |
|------|-------------|---------------|----------|
| **XGBoost** | **0.6429** | **0.6425** | **0.7004** |
| LightGBM | 0.6275 | 0.6469 | 0.6945 |
| ExtraTrees | 0.6253 | 0.6118 | 0.6851 |
| Random Forest | 0.6181 | 0.6162 | 0.6883 |
| Stacking Ensemble | N/A | **0.5855** | **0.6711** |

*(Metrics auto-update in `outputs/results/model_results.json` upon running `--mode train`)*

---

## 📁 Source Code Structure

```text
IPL-Winner-Prediction/
├── main.py                     # Primary pipeline controller
├── convert_cricsheet.py        # 🆕 Ingestion script for massive Cricsheet data
├── config.py                   # Global constants and Expected XIs
├── data/                       # Contains isolated raw/processed artifacts + SQLite DB
├── src/
│   ├── data/create_dataset.py  # 🛠️ Builds Match Rosters & Phase-Stats
│   ├── features/
│   │   ├── engineer.py         # 📈 Elo Rating Engine
│   │   ├── team_strength.py    # 🏏 Phase-Specific Bat/Bowl strength
│   │   └── venue_features.py   # 🏟️ Rolling Venue Conditions logic
│   ├── models/                 # Random Forest, XGBoost, Ensemble layers
│   └── prediction/
│       ├── match_predictor.py  # 🆚 Interactive Match Simulator
│       └── predict_2026.py     # Global Franchise projections
└── outputs/                    # Auto-generated visual artifacts and trained PKLs
```

## 📝 License
MIT License. Feel free to fork, expand upon this data engineering pipeline, and apply it to other formats!
