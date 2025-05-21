# This script generates explanations for the top 3 property recommendations using
# SHAP values and GPT-3.5 to provide human-readable insights about why each
# property was selected as a good comparable.



# 1. Imports & Environment Setup
#    - shap: model interpretability
#    - xgboost: load trained model
#    - pandas: data manipulation
#    - numpy: numerical operations
#    - openai: GPT API access
#    - dotenv: environment variable loading
#    - os: environment variable access

import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os



# 2. Model & Data Loading
#    - Loads trained XGBoost model
#    - Loads training data
#    - Defines feature columns

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")
print(f"Using API key starting with: {api_key[:5]}")

client = OpenAI(api_key=api_key)

# Load model and dataset
model = xgb.Booster()
model.load_model("../models/xgb_rank_model.json")
df = pd.read_csv("../data/training/training_data.csv")

# Feature columns used during training
feature_cols = [
    'bath_score_diff', 'full_baths_diff', 'half_baths_diff',
    'room_count_diff', 'bedrooms_diff', 'effective_age_diff',
    'subject_age_diff', 'lot_size_sf_diff', 'gla_diff',
    'abs_bath_score_diff', 'abs_full_bath_diff', 'abs_half_bath_diff',
    'abs_room_count_diff', 'abs_bedrooms_diff', 'abs_effective_age_diff',
    'abs_subject_age_diff', 'abs_lot_size_sf_diff', 'abs_gla_diff',
    'same_property_type', 'sold_recently'
]



# 3. GPT Explanation Generation
#    - gpt_explanation function:
#      • Takes model score and SHAP values
#      • Formats positive/negative factors
#      • Calls GPT-3.5 to generate explanation
#      • Returns human-readable text

# GPT explanation function
def gpt_explanation(score, pos_feats, neg_feats, candidate_address, subject_address):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a real estate appraisal assistant. Your job is to explain why a specific property "
                        "is or isn't a good comparable to a given subject property, based on feature-level model evaluation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"You're evaluating whether the property at **{candidate_address}** is a good comparable "
                        f"for the subject property located at **{subject_address}**.\n\n"
                        f"The model gave this candidate a score of {score:.2f}.\n\n"
                        f"These features helped it rank higher:\n"
                        f"{', '.join([f'{f} ({v:.2f})' for f, v in pos_feats]) or 'None'}\n\n"
                        f"These features hurt its ranking:\n"
                        f"{', '.join([f'{f} ({v:.2f})' for f, v in neg_feats]) or 'None'}\n\n"
                        f"Write an brief explanation (1–2 sentences) of whether this is a good comparable and why."
                    )
                }
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error getting GPT explanation: {e}]"



# 4. SHAP Analysis
#    - Creates SHAP explainer for model
#    - For each appraisal:
#      • Predicts scores for all properties
#      • Takes top 3 by score
#      • Computes SHAP values for each
#      • Identifies positive/negative factors

# Ensure data types are float for SHAP compatibility
df[feature_cols] = df[feature_cols].astype(float)

def model_predict(X_df):
    dmatrix = xgb.DMatrix(X_df)
    return model.predict(dmatrix)

explainer = shap.Explainer(model_predict, df[feature_cols])



# 5. Results Generation
#    - For each top 3 property:
#      • Gets GPT explanation
#      • Stores orderID, address, score, explanation

results = []
for i, (order_id, group) in enumerate(df.groupby("orderID")):
    group = group.copy()
    group[feature_cols] = group[feature_cols].astype(float)
    dmatrix = xgb.DMatrix(group[feature_cols])
    group["score"] = model.predict(dmatrix)
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

        print(f"→ Sending GPT request for orderID={order_id}, address={row['candidate_address']}")
        explanation = gpt_explanation(row['score'], positive_factors[:3], negative_factors[:3], row['candidate_address'], row['subject_address'])

        results.append({
            "orderID": order_id,
            "candidate_address": row["candidate_address"],
            "score": row["score"],
            "explanation": explanation
        })



# 6. Output
#    - Saves results to CSV
#    - Prints first 10 explanations

# Save results
top3_df = pd.DataFrame(results)
top3_df.to_csv("../outputs/top3_gpt_explanations.csv", index=False)
print(top3_df.head(10))



# Output: "top3_gpt_explanations.csv" containing:
#         - orderID: appraisal identifier
#         - candidate_address: recommended property
#         - score: model's ranking score
#         - explanation: GPT-generated explanation