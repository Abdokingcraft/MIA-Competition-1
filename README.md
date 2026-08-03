# Timeline

## Day 1 :

i started by watching "intro to machine learning" from the discord and then i watched a ML video on youtube on how to implement it to python [here](https://www.youtube.com/watch?v=7eh4d6sabA0) (`DecisionTreeClassifier`) but it didn't work with the prediction that i want to get, it only could predict strings (cat, not a cat , dog , not a dog , etc), so i found another ML method called (`DecisionTreeRegressor`) from this [youtuber](https://www.youtube.com/watch?v=08QNwMQ0Mg8) and with the help of AI i finally got a code that could work, the first predection got 0.56, so i decided to remove some unwanted feature (hight, preferred_foot , etc) and it got me ..... 0.56 still

that is when i relized i didn't save it, so when i tried again i got ....... 0.27 :D

i forgot to use 100% of the train data for training, so after everything i got 0.566

overall, i dont think DTC (`DecisionTreeClassifier`) is the best fit for this task

---

## Day 2 :

i just noticed that having a lower score is better than having a higher score 🫠 , now i am trying to figure out what did i do to get that 0.27 🤠
I started looking for ways that I can improve my code but I haven't done anything that day

---

## Day 3 :

I also haven't done anything that day on the competition because I was focusing on finishing Task 2 that I couldn't do

---

## Day 4 :

i started thinking about 2 things

1. having multiple models
  
2. cutting down features or adding them all
  

so i grabbed most of the models from sklearn so i could see which one would fit best :

**LassoLars** : `LassoLars`

**BayesianRidge** : `BayesianRidge`

**ARD** : `ARDRegression`

**SGD** : `SGDRegressor`

**Huber** : `HuberRegressor`

**RANSAC** : `RANSACRegressor`

**TheilSen** : `TheilSenRegressor`

**DT** : `DecisionTreeRegressor`

**RF** : `RandomForestRegressor`

**ET** : `ExtraTreesRegressor`

**Bagging** : `BaggingRegressor`

**AdaBoost** : `AdaBoostRegressor`

**GBR** : `GradientBoostingRegressor`

**HistGBR** : `HistGradientBoostingRegressor`

**KNN** : `KNeighborsRegressor`

**MLP** : `MLPRegressor`

i looked into most of the models before i did any coding, i decided to start with GBR for testing how models work

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(
    n_estimators=200,    # number of trees      
    learning_rate=0.1,   # learning speed    
    max_depth=4,         # trees' depth    
    subsample=0.8        # having each tree explain 80% of the data
)
```

these are the parameters that is used to help the model

so i made a python code which loops though some set parameters and give me the result

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

feature_cols = [
    "age", "minutes_played", "goals", "assists", "shots", "shots_on_target",
    "expected_goals_xg", "expected_assists_xa", "key_passes",
    "successful_passes", "pass_accuracy", "dribbles_attempted", 
    "successful_dribbles", "crosses", "successful_crosses",
    "tackles", "interceptions", "clearances", "blocks",
    "aerial_duels_won", "aerial_duels_lost", "recoveries", "defensive_actions",
    "fouls_committed", "yellow_cards", "red_cards", "offsides",
    "saves", "save_percentage", "punches", "clean_sheet", "goals_conceded", 
    "penalty_saves", "distance_covered_km", "sprint_distance_km", "top_speed_kmh", 
    "stamina_score", "offensive_contribution", "defensive_contribution", 
    "possession_impact", "pressure_resistance", "creativity_score", 
    "consistency_score", "clutch_performance_score",
    "total_goals_tournament", "total_assists_tournament", "total_minutes_tournament",
    "player_of_match_awards"
]

print("Loading data...")
train_df = pd.read_csv("tasks3/train.csv")
test_df = pd.read_csv("tasks3/test.csv")
solution_df = pd.read_csv("tasks3/solution.csv")

test_df = test_df.merge(solution_df, on="Id")
test_df.rename(columns={'player_rating': 'actual_rating'}, inplace=True)

X_train = train_df[feature_cols]
y_train = train_df['player_rating']

X_test = test_df[feature_cols]
y_test = test_df['actual_rating']


for n in range(1,11):
    for l in range(5,50,5):
        for d in range(10,100,10):
            for sub in range(1,10):



                gbr_model = GradientBoostingRegressor(
                    n_estimators=100 * n,        
                    learning_rate=0.001 * l,       
                    max_depth= d,            
                    subsample=0.1 * sub,   
                    random_state=42
                )

                gbr_model.fit(X_train, y_train)
                predictions = gbr_model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)

                print("-" * 30)
                print(f"(MSE): {mse:.4f} for n = {n*100}, l = {0.001*l}, depth = {d*40}, sub = {0.1 *sub}")
                print("-" * 30)
```

it took sometime but i found the best one

- subsample: 0.7
- n_estimators: 300
- max_depth: 5
- learning_rate: 0.025

it got me 0.234

though weirdly, when i submitted it on kaggle it showed 0.275 instead of 0.234 locally, i think it might be a public/private leaderboard split thing but i'm not 100% sure

---

## Day 5 :

nothing happened this day, i was busy learning about neural networks instead

---

## Day 6 :

same as day 5, still nothing on the competition, still learning about neural networks

---

## Day 7 (last day) :

i scrapped GBR and switched to `HistGradientBoostingRegressor` instead

before switching, i went back and actually tested every model from my day 4 list with a script AI helped me write, just to see which one performed best on this data, turns out GBR and HistGBR were the best two out of all of them

it was the last day so i didn't really have anything else left to try, turned out to be a good call though, i used it and it got me ~0.24 something, finished 5th place with it 🎉 (hopefully , there is still 20 min left at the time of typing this)

i also added a small manual rule on top of the model: if a player's `minutes_played` is 0, their predicted rating gets forced to 0.0 instantly instead of letting the model guess a prediction for them

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Data
train_df = pd.read_csv("tasks3/train.csv")
test_df = pd.read_csv("tasks3/test.csv")
solution_df = pd.read_csv("tasks3/solution.csv")

# 2. Complete Feature Selection
feature_cols = [
    "position", "preferred_foot", "match_result",
    "market_value_eur", # Note: Usually named 'market_value_eur' in your earlier data, check your column name
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

# 3. Preprocessing & Encoding
cat_cols = [col for col in ["position", "preferred_foot", "match_result"] if col in feature_cols]

X_train = train_df[feature_cols].copy()
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
y_train = train_df['player_rating']

X_test = test_df[feature_cols].copy()
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

test_with_solution = test_df.merge(solution_df, on="Id")
y_test = test_with_solution['player_rating']

# 4. Define & Train Model
# Using HistGradientBoostingRegressor to avoid NaNs crashing the script

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

# 5. Predictions & Evaluation
predictions = gbr_model.predict(X_test)

# 6. Apply the 0-minute rule to predictions
# If a player played 0 minutes, force their rating to 0.0
zero_mask = test_df["minutes_played"] == 0
predictions[zero_mask] = 0.0

mse = mean_squared_error(y_test, predictions)
print(f"(MSE): {mse:.4f}")

# 7. Export Predictions
submission_df = pd.DataFrame({
    "Id": test_df["Id"],
    "player_rating": predictions
})
submission_df.to_csv("tasks3/submission.csv", index=False)
```
