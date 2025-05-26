# This script trains an XGBoost ranking model to identify good comparable properties
# using pairwise ranking objective. It supports feedback-driven training data and
# evaluates model performance using Top-K precision metrics (K = 1, 3).



# 1. Imports & Constants
#    - pandas: for data manipulation
#    - xgboost: for training the ranking model
#    - sklearn.model_selection: for stratified train-test split
#    - numpy: for label shuffling (optional sanity check)
#    - os: to check file existence and size
#    - SHUFFLE_LABELS: flag to randomize comp labels for sanity debugging

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import os

SHUFFLE_LABELS = False
TRAINING_DATA_FILE = "../data/training/training_data_with_feedback.csv" if os.path.exists("../frontend/feedback/feedback_log.csv") and os.path.getsize("../frontend/feedback/feedback_log.csv") > 0 else "../data/training/training_data.csv"



# 2. Data Loading & Label Shuffling
#    - Chooses training dataset based on whether feedback_log.csv is present and non-empty
#    - Optionally shuffles is_comp labels within each orderID group for sanity testing
#    - Creates `label` column from is_comp

df = pd.read_csv(TRAINING_DATA_FILE)

print(f"Using training data: {TRAINING_DATA_FILE}")

if SHUFFLE_LABELS:
    print("Shuffling labels for sanity check...")
    df["is_comp"] = df.groupby("orderID")["is_comp"].transform(
        lambda x: np.random.permutation(x.values)
    )



# 3. Feature Columns
#    - Defines list of features to be used for training
#    - Includes raw and absolute difference metrics, plus recency and type match flags

feature_cols = [
    'bath_score_diff', 'full_baths_diff', 'half_baths_diff',
    'room_count_diff', 'bedrooms_diff', 'effective_age_diff',
    'subject_age_diff', 'lot_size_sf_diff', 'gla_diff',
    'abs_bath_score_diff', 'abs_full_bath_diff', 'abs_half_bath_diff',
    'abs_room_count_diff', 'abs_bedrooms_diff', 'abs_effective_age_diff',
    'abs_subject_age_diff', 'abs_lot_size_sf_diff', 'abs_gla_diff',
    'same_property_type', 'sold_recently', # 'distance_to_subject_km'
]

# Fill in label if not already present
df['label'] = df['is_comp']



# 4. Train-Test Split
#    - Splits data into training and test sets using 80/20 ratio
#    - Stratifies by label to preserve comp/non-comp ratio
#    - Sorts by orderID to prepare for group-based ranking

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# Sort for group creation
df_train = df_train.sort_values("orderID")
df_test = df_test.sort_values("orderID")



# 5. DMatrix Preparation
#    - Groups training data by orderID (used for ranking)
#    - Converts features to float and constructs XGBoost DMatrix
#    - Sets group sizes for pairwise ranking model

# Group by orderID for ranking
groups_train = df_train.groupby("orderID").size().to_list()

# Ensure numeric input (float) for DMatrix
X_train = df_train[feature_cols].astype(float)
y_train = df_train["label"]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(groups_train)



# 6. Model Training
#    - Defines XGBoost ranking parameters:
#        • rank:pairwise objective
#        • ndcg evaluation metric
#        • learning rate (eta) = 0.1
#        • tree depth = 6
#    - Trains model using 100 boosting rounds

params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'eta': 0.1,
    'max_depth': 6,
    'verbosity': 1
}

model = xgb.train(params, dtrain, num_boost_round=100)



# 7. Model Evaluation
#    - Defines evaluate_topk(df_group, k) to:
#        • Predict scores for each group
#        • Select top K candidates by score
#        • Count how many are actually comps (label = 1)
#    - Evaluates and prints Top-1 and Top-3 precision

print("\nTop-K Evaluation by Appraisal:")

def evaluate_topk(df_group, k=3):
    df_group = df_group.copy()
    X = xgb.DMatrix(df_group[feature_cols].astype(float))
    df_group["score"] = model.predict(X)
    topk = df_group.sort_values("score", ascending=False).head(k)
    correct = topk["label"].sum()
    return pd.Series({"correct": correct, "total": k})

# Evaluate at K = 1, 3
for k in [1, 3]:
    results = df_test.groupby("orderID").apply(lambda g: evaluate_topk(g, k)).sum()
    precision = results["correct"] / results["total"]
    print(f"Top-{k} Precision: {precision:.3f}")



# 8. Model Saving
#    - Saves trained model as "xgb_rank_model.json" for future inference

model.save_model("../models/xgb_rank_model.json")
print("\nRanking model saved as ../models/xgb_rank_model.json")