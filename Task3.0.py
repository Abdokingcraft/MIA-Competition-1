from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error



CSV_PATH = Path(__file__).resolve().parent / "archive"


# Loading Data
train_df = pd.read_csv(str(CSV_PATH / "train.csv"))
test_df = pd.read_csv(str(CSV_PATH / "test.csv"))
solution_df = pd.read_csv(str(CSV_PATH / "solution.csv"))


feature_cols = [
    "position", "preferred_foot", "match_result",
    "market_value_eur", 
    "goals_team", "goals_opponent",
    "minutes_played", "goals", "assists", "shots", "shots_on_target",
    "expected_goals_xg", "expected_assists_xa", "key_passes",
    "successful_passes", "total_passes", "pass_accuracy",
    "dribbles_attempted", "successful_dribbles", "crosses", "successful_crosses",
    "tackles", "interceptions", "clearances", "blocks",
    "aerial_duels_won", "aerial_duels_lost", "recoveries", "defensive_actions",
    "fouls_committed", "fouls_suffered", "yellow_cards", "red_cards", "offsides",
    "saves", "save_percentage", "punches", "clean_sheet", "goals_conceded", "penalty_saves",
    "distance_covered_km", "sprint_distance_km", "top_speed_kmh",
    "accelerations", "decelerations", "stamina_score",
    "offensive_contribution", "defensive_contribution",
    "possession_impact", "pressure_resistance", "creativity_score",
    "consistency_score", "clutch_performance_score",
    "total_goals_tournament", "total_assists_tournament", "total_minutes_tournament",
    "player_of_match_awards"
]

feature_cols = [col for col in feature_cols if col in train_df.columns]


cat_cols = [col for col in ["position", "preferred_foot", "match_result"] if col in feature_cols]

X_train = train_df[feature_cols].copy()
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
y_train = train_df['player_rating']

X_test = test_df[feature_cols].copy()
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

test_with_solution = test_df.merge(solution_df, on="Id")
y_test = test_with_solution['player_rating']

# Training model

gbr_model = HistGradientBoostingRegressor(
    max_iter=1000,             # More trees
    max_depth=7,               # Slightly deeper
    learning_rate=0.025,       # Slower learning
    l2_regularization=2.0,     # Stronger penalty on complex trees
    min_samples_leaf=30,       # Prevents overfitting on outlier players
    random_state=6767
)

print("Training model with expanded feature set...")
gbr_model.fit(X_train, y_train)


predictions = gbr_model.predict(X_test)

# If a player played 0 minutes, force their rating to 0.0
zero_mask = test_df["minutes_played"] == 0
predictions[zero_mask] = 0.0

mse = mean_squared_error(y_test, predictions)
print(f"(MSE): {mse:.4f}")

submission_df = pd.DataFrame({
    "Id": test_df["Id"],
    "player_rating": predictions
})
submission_df.to_csv(str(CSV_PATH / "submission.csv"), index=False)