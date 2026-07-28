import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR

feature_cols = [
    "position", "age", "preferred_foot",
    "minutes_played", "goals", "assists", "shots", "shots_on_target",
    "expected_goals_xg", "expected_assists_xa", "key_passes",
    "successful_passes", "pass_accuracy",
    "dribbles_attempted", "successful_dribbles", "crosses", "successful_crosses",
    "tackles", "interceptions", "clearances", "blocks",
    "aerial_duels_won", "aerial_duels_lost", "recoveries", "defensive_actions",
    "fouls_committed", "yellow_cards", "red_cards", "offsides",
    "saves", "save_percentage", "punches", "clean_sheet", "goals_conceded", "penalty_saves",
    "distance_covered_km", "sprint_distance_km", "top_speed_kmh", "stamina_score"
]

# --- training (same as before) ---
df = pd.read_csv("tasks3/train.csv")
enput = df[feature_cols]
output = df["player_rating"]
enput = pd.get_dummies(enput, columns=["position", "preferred_foot"])

model = DTR()
model.fit(enput, output)

# --- predicting on the test set ---
test_df = pd.read_csv("tasks3/test.csv")

player_ids = test_df["Id"]          # keep this aside, NOT a feature
X_test = test_df[feature_cols]
X_test = pd.get_dummies(X_test, columns=["position", "preferred_foot"])
X_test = X_test.reindex(columns=enput.columns, fill_value=0)

predictions = model.predict(X_test)

# --- building the output: player_id first, then the predicted rating ---
results = pd.DataFrame({
    "Id": player_ids,
    "player_rating": predictions
})

results.to_csv("tasks3/predictions.csv", index=False)
print(results.head())