# This script generates explanations for the top 3 property recommendations per appraisal
# by combining SHAP values (feature importance) with GPT-generated natural language summaries.
# It uses the trained XGBoost model and the feature-engineered dataset.



# 1. Imports & Setup
#    - shap: computes feature attributions (SHAP values)
#    - xgboost: loads trained ranking model
#    - pandas/numpy: data manipulation
#    - openai: GPT-based explanation generation
#    - os: environment access
#    - tqdm: progress bar for batch processing
#    - json: load raw appraisal data for enrichment

import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from tqdm import tqdm
import json



# 2. Environment & Model Loading
#    - Loads OpenAI API key and initializes client
#    - Loads trained XGBoost model from disk
#    - Loads raw appraisal JSON and feature-engineered CSV (with or without feedback)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=api_key)

# Load the trained model
model = xgb.Booster()
model.load_model("../models/xgb_rank_model.json")

with open("../data/engineered/feature_engineered_appraisals_dataset.json") as f:
    raw_data = json.load(f)

data_file = (
    "../data/training/training_data_with_feedback.csv"
    if os.path.exists("../frontend/feedback/feedback_log.csv") and os.path.getsize("../frontend/feedback/feedback_log.csv") > 0
    else "../data/training/training_data.csv"
)

df = pd.read_csv(data_file)
print(f"Using training data: {data_file}")



# 3. Feature Configuration
#    - Defines list of feature columns used for training/inference
#    - Ensures these columns are float-typed for XGBoost/SHAP compatibility

feature_cols = [
    'bath_score_diff', 'full_baths_diff', 'half_baths_diff',
    'room_count_diff', 'bedrooms_diff', 'effective_age_diff',
    'subject_age_diff', 'lot_size_sf_diff', 'gla_diff',
    'abs_bath_score_diff', 'abs_full_bath_diff', 'abs_half_bath_diff',
    'abs_room_count_diff', 'abs_bedrooms_diff', 'abs_effective_age_diff',
    'abs_subject_age_diff', 'abs_lot_size_sf_diff', 'abs_gla_diff',
    'same_property_type', 'sold_recently', # 'distance_to_subject_km'
]

df[feature_cols] = df[feature_cols].astype(float)



# 4. Raw Data Enrichment: find_raw_values(order_id, candidate_address)
#    - Extracts subject and candidate attributes from raw JSON
#    - Returns a dictionary with extra details for inclusion in GPT explanations

# Lookup actual property info 
def find_raw_values(order_id, candidate_address):
    for appraisal in raw_data["appraisals"]:
        if str(appraisal.get("orderID")) != str(order_id):
            continue
        subject = appraisal.get("subject", {})
        subject_vals = {
            "subject_bath_score": subject.get("bath_score"),
            "subject_num_full_baths": subject.get("num_full_baths"),
            "subject_num_half_baths": subject.get("num_half_baths"),
            "subject_bedrooms": subject.get('num_beds'),
            "subject_gla": subject.get("gla"),
            "subject_lot_size_sf": subject.get("lot_size_sf"),
            "subject_property_type": subject.get("property_type"),
        }
        for group in ("comps", "properties"):
            for prop in appraisal.get(group, []):
                if prop.get("address", "").strip().lower() == candidate_address.strip().lower():
                    return subject_vals | {
                        "candidate_bath_score": prop.get("bath_score"),
                        "candidate_num_full_baths": prop.get("num_full_baths"),
                        "candidate_num_half_baths": prop.get("num_half_baths"),
                        "candidate_bedrooms": prop.get('num_beds'),
                        "candidate_gla": prop.get("gla"),
                        "candidate_lot_size_sf": prop.get("lot_size_sf"),
                        "candidate_property_type": prop.get("property_type"),
                        "candidate_close_price": prop.get("sale_price")
                    }
    return subject_vals



# 5. GPT Explanation: gpt_explanation(...)
#    - Formats SHAP results into readable strings with actual values
#    - Calls GPT-3.5 to generate a 1–2 sentence explanation for a candidate's ranking score
#    - System prompt defines a role-specific assistant trained for real estate comparisons

def gpt_explanation(score, pos_feats, neg_feats, candidate_address, subject_address):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        """You are a real estate appraisal assistant. Your job is to explain why a machine learning model ranked a candidate property as more or less comparable to a subject property.

                            The model uses feature differences (e.g., size difference, age difference) between the candidate and subject. Positive SHAP values mean the feature made the 
                            candidate more similar (better match), while negative SHAP values indicate dissimilarity.

                            Do not say whether the property is 'good' or 'bad'. Instead, explain how the model interpreted the feature similarities or differences that affected the score."""
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"""
                        The model assigned the candidate property at {candidate_address} a score of {score:.2f} when comparing it to the subject at {subject_address}.

                        Features that made the candidate more similar:
                        {', '.join([f'{f} ({v:.2f})' for f, v in pos_feats]) or 'None'}

                        Features that made the candidate less similar:
                        {', '.join([f'{f} ({v:.2f})' for f, v in neg_feats]) or 'None'}

                        Based only on the features and their impact, describe why the model produced this score. Focus on the similarity or difference in attributes.
                        """
                    )
                }
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error getting GPT explanation: {e}]"

def gpt_explanation(score, pos_feats, neg_feats, candidate_address, subject_address, row):
    def enrich(features):
        return ', '.join(
            f"{f} = {row.get(f, 'N/A')} (SHAP {v:+.2f})"
            for f, v in features
        )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a real estate appraisal assistant. Your job is to explain why a machine learning model ranked a candidate property as more or less comparable to a subject property.\n\n"
                        "The model uses feature differences (e.g., size difference, age difference) between the candidate and subject. Positive SHAP values mean the feature made the candidate more similar (better match), while negative SHAP values indicate dissimilarity.\n\n"
                        "Do not say whether the property is 'good' or 'bad'. Instead, explain how the model interpreted the feature similarities or differences that affected the score. Use both the actual feature values and their SHAP impact scores."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"The model gave the candidate property at {candidate_address} a score of {score:.2f} when comparing it to the subject at {subject_address}.\n\n"
                        f"These features made the candidate more similar:\n{enrich(pos_feats) or 'None'}\n\n"
                        f"These features made the candidate less similar:\n{enrich(neg_feats) or 'None'}\n\n"
                        "Using the actual values and SHAP scores, explain in 1–2 sentences why the model ranked this candidate where it did."
                    )
                }
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error getting GPT explanation: {e}]"



# 6. SHAP Wrapper Setup
#    - Defines `model_predict` for SHAP compatibility with XGBoost
#    - Initializes SHAP Explainer over the full training set

# SHAP wrapper  
def model_predict(X_df):
    dmatrix = xgb.DMatrix(X_df)
    return model.predict(dmatrix)

explainer = shap.Explainer(model_predict, df[feature_cols])



# 7. Top-3 Generation & Explanation Loop
#    - For each appraisal group:
#        • Predicts model scores
#        • Ranks and selects top 3 candidates
#        • Computes SHAP values
#        • Extracts most positive/negative contributing features
#        • Enriches row with raw values from JSON
#        • Gets GPT-generated explanation and stores result

# Main loop 
results = []
for order_id, group in tqdm(df.groupby("orderID"), desc="Generating GPT Explanations"):
    group = group.copy()
    group[feature_cols] = group[feature_cols].astype(float)
    dmatrix = xgb.DMatrix(group[feature_cols])
    group["score"] = model.predict(dmatrix)
    group["rank"] = group["score"].rank(method="first", ascending=False)

    top3 = group.sort_values("score", ascending=False).head(3)

    for _, row in top3.iterrows():
        row_df = row[feature_cols].to_frame().T.astype(float)
        try:
            shap_vals = explainer(row_df)
        except Exception as e:
            print(f"[SHAP Error] orderID={order_id}: {e}")
            continue

        shap_items = list(zip(row_df.columns, shap_vals.values[0]))
        positive_factors = [(f, v) for f, v in shap_items if v > 0]
        negative_factors = [(f, v) for f, v in shap_items if v < 0]

        extra = find_raw_values(order_id, row["candidate_address"])
        enriched_row = row.to_dict() | extra | {"orderID": order_id}

        explanation = gpt_explanation(
            row['score'], positive_factors[:3], negative_factors[:3],
            row["candidate_address"], row["subject_address"], enriched_row
        )

        enriched_row["explanation"] = explanation
        results.append(enriched_row)



# 8. Save Output
#    - Converts all results into a DataFrame
#    - Saves to CSV ("top3_gpt_explanations.csv")
#    - Prints summary stats: total rows, number of true comps, precision, false positives

# Final output 
top3_df = pd.DataFrame(results)
top3_df = top3_df.sort_values(by=["orderID", "score"], ascending=[True, False])
top3_df.to_csv("../outputs/top3_gpt_explanations.csv", index=False)
print("\nSaved top3_gpt_explanations.csv to outputs directory")

# Analysis 
print("\n[Results Analysis]")
print("Total top-3 rows:", len(top3_df))
print("How many are labeled comps (is_comp = 1)?", top3_df["is_comp"].sum())
print("Top-3 Precision:", top3_df["is_comp"].mean())
print(top3_df["is_comp"].value_counts())

false_positives = top3_df[top3_df["is_comp"] == 0][["orderID", "candidate_address"]]
print("\nFalse Positives (Top-3 predicted but not comps):")
print(false_positives.to_string(index=False))