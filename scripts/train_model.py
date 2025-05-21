# This script trains an XGBoost ranking model to identify good comparable properties
# using pairwise ranking objective. It evaluates the model's performance using
# precision at different K values (1, 3, 5).



# 1. Imports & Constants
#    - pandas: data manipulation
#    - xgboost: gradient boosting for ranking
#    - sklearn: train-test split
#    - numpy: numerical operations

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np



# 2. Data Loading & Preparation
#    - Loads training data CSV
#    - Defines feature columns used for training
#    - Creates label from is_comp field

df = pd.read_csv("../data/training/training_data.csv")

feature_cols = [
    'bath_score_diff', 'full_baths_diff', 'half_baths_diff',
    'room_count_diff', 'bedrooms_diff', 'effective_age_diff',
    'subject_age_diff', 'lot_size_sf_diff', 'gla_diff',
    'abs_bath_score_diff', 'abs_full_bath_diff', 'abs_half_bath_diff',
    'abs_room_count_diff', 'abs_bedrooms_diff', 'abs_effective_age_diff',
    'abs_subject_age_diff', 'abs_lot_size_sf_diff', 'abs_gla_diff',
    'same_property_type', 'sold_recently'
]

df['label'] = df['is_comp']



# 3. Train-Test Split
#    - Splits data 80/20 stratified by label
#    - Sorts by orderID for group creation

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)



# 4. Model Training
#    - Creates XGBoost DMatrix with groups
#    - Sets ranking parameters:
#      • objective: rank:pairwise
#      • eval_metric: ndcg
#      • eta: 0.1 (learning rate)
#      • max_depth: 6
#    - Trains for 100 boosting rounds

# Sort for group creation
df_train = df_train.sort_values("orderID")
df_test = df_test.sort_values("orderID")

# Group by orderID for ranking
groups_train = df_train.groupby("orderID").size().to_list()

# Ensure numeric input (float) for DMatrix
X_train = df_train[feature_cols].astype(float)
y_train = df_train["label"]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(groups_train)

# Train ranking model
params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'eta': 0.1,
    'max_depth': 6,
    'verbosity': 1
}

model = xgb.train(params, dtrain, num_boost_round=100)



# 5. Model Evaluation
#    - evaluate_topk function:
#      • Predicts scores for properties in group
#      • Takes top K by score
#      • Returns precision (correct/total)
#    - Evaluates at K=1,3,5

print("\nTop-K Evaluation by Appraisal:")

def evaluate_topk(df_group, k=3):
    df_group = df_group.copy()
    X = xgb.DMatrix(df_group[feature_cols].astype(float))
    df_group["score"] = model.predict(X)
    topk = df_group.sort_values("score", ascending=False).head(k)
    correct = topk["label"].sum()
    return pd.Series({"correct": correct, "total": k})

# Evaluate at K = 1, 3, 5
for k in [1, 3, 5]:
    results = df_test.groupby("orderID").apply(lambda g: evaluate_topk(g, k)).sum()
    precision = results["correct"] / results["total"]
    print(f"Top-{k} Precision: {precision:.3f}")



# 6. Model Saving
#    - Saves trained model to JSON file

model.save_model("../models/xgb_rank_model.json")
print("\nRanking model saved as xgb_rank_model.json")



# Output: "xgb_rank_model.json" containing the trained model
#         Prints precision metrics for different K values