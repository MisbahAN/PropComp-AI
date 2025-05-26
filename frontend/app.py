# 1. Imports & File Paths
#    - streamlit: frontend web interface
#    - pandas: CSV reading and manipulation
#    - os: file path and existence checks
#    - subprocess: rerun backend scripts upon feedback

import streamlit as st
import pandas as pd
import os
import subprocess

# File locations
EXPLANATIONS_FILE = "../outputs/top3_gpt_explanations.csv"
FEEDBACK_FILE = "feedback/feedback_log.csv"



# 2. Load GPT-generated explanations

df = pd.read_csv(EXPLANATIONS_FILE)

# Extract appraisal IDs for dropdown
order_ids = sorted(df["orderID"].unique())
selected_order = st.selectbox("Select an Appraisal (orderID)", order_ids)

# Filter data for selected appraisal
appraisal_df = df[df["orderID"] == selected_order].sort_values("rank")

# UI Headers
st.title("🏠 Property Comparison Feedback")
st.subheader(f"Subject Property: {appraisal_df['subject_address'].iloc[0]}")
st.markdown("---")

# Placeholder to store user feedback
feedback_records = []



# 3. Helper functions for formatting values

def format_int(val):
    try:
        return int(round(float(val)))
    except:
        return "Not available"

def format_price(val):
    try:
        return f"${int(round(float(val))):,}"
    except:
        return "Not available"



# 4. Iterate over top-3 candidate properties and collect feedback
valid_prices = []

for _, row in appraisal_df.iterrows():
    # Display candidate details
    st.markdown(f"### 🏘️ Candidate Property (Rank {int(row['rank'])}):")
    st.markdown(f"**Address:** {row['candidate_address']}")
    st.markdown(f"**Model Score:** `{row['score']:.2f}`")
    st.markdown(f"**Explanation:** {row['explanation']}")

     # Create side-by-side feature comparison table
    st.markdown("#### 📊 Feature Comparison")
    comparison_data = {
        "Feature": [
            "Bedrooms", "Full Bathrooms", "Half Bathrooms",
            "GLA (sq ft)", "Lot Size (sq ft)",
            "Property Type"
        ],
        "Subject": [
            format_int(row.get("subject_bedrooms")),
            format_int(row.get("subject_num_full_baths")),
            format_int(row.get("subject_num_half_baths")),
            format_int(row.get("subject_gla")),
            format_int(row.get("subject_lot_size_sf")),
            row.get('subject_property_type') or "Not available"
        ],
        "Candidate": [
            format_int(row.get("candidate_bedrooms")),
            format_int(row.get("candidate_num_full_baths")),
            format_int(row.get("candidate_num_half_baths")),
            format_int(row.get("candidate_gla")),
            format_int(row.get("candidate_lot_size_sf")),

            row.get('candidate_property_type') or "Not available"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data).astype(str)
    st.table(comparison_df)

    # Show candidate closing price
    close_price = row.get("candidate_close_price")
    st.markdown(f"**Candidate Close Price:** {format_price(close_price)}")

    # Save for price suggestion calculation
    try:
        valid_prices.append(float(close_price))
    except:
        pass

    # Collect user feedback
    key = f"feedback_{row['orderID']}_{row['rank']}"
    feedback = st.radio("Do you agree this is a good comparable?", ("👍 Yes", "👎 No"), key=key)

    feedback_records.append({
        "orderID": row["orderID"],
        "rank": row["rank"],
        "candidate_address": row["candidate_address"],
        "subject_address": row["subject_address"],
        "score": row["score"],
        "is_comp": row["is_comp"],
        "user_feedback": 1 if feedback == "👍 Yes" else 0
    })

st.markdown("---")



# 5. Show suggested price estimate based on top-3 comps
st.header("💰 Suggested Value Estimate")

if valid_prices:
    avg_price = sum(valid_prices) / len(valid_prices)
    min_price = min(valid_prices)
    max_price = max(valid_prices)
    mid_point = min_price + ((max_price-min_price) / 2)

    # Display estimate metrics with formatting
    st.markdown(
        f"""
        <div style='margin-top: 1rem;'>
            <span style='font-size: 1.15rem; font-weight: 600;'>Average Price of Top-3 Comps:</span>
            <span style='font-size: 1.15rem; font-weight: 500; margin-left: 0.5rem;'>
                {format_price(avg_price)}
            </span>
        </div>
        <div style='margin-top: 1rem;'>
            <span style='font-size: 1.15rem; font-weight: 600;'>Suggested Price Range:</span>
            <span style='font-size: 1.15rem; font-weight: 500; margin-left: 0.5rem;'>
                {format_price(min_price)} - {format_price(max_price)}
            </span>
        </div>
        <div style='margin-top: 1rem;'>
            <span style='font-size: 1.15rem; font-weight: 600;'>Suggested Price Range Midpoint:</span>
            <span style='font-size: 1.15rem; font-weight: 500; margin-left: 0.5rem;'>
                {format_price(mid_point)}
            </span>
        </div>
        <div style='margin-top: 1rem; margin-bottom: 1rem'>
            <span style='font-size: 0.8rem; font-weight: 600; color: grey'>
                This estimate is based on the closing prices of the top 3 comparable properties selected by the model.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown("Not enough valid close price data to calculate a suggested value.")



# 6. Save feedback and trigger pipeline retraining

if st.button("✅ Submit Feedback"):
    feedback_df = pd.DataFrame(feedback_records)

    if os.path.exists(FEEDBACK_FILE):
        try:
            existing = pd.read_csv(FEEDBACK_FILE)

            # Drop duplicates by orderID and candidate_address
            combined = pd.concat([existing, feedback_df])
            combined = combined.drop_duplicates(
                subset=["orderID", "candidate_address"], keep="last"
            )
            combined.to_csv(FEEDBACK_FILE, index=False)
        except pd.errors.EmptyDataError:
            feedback_df.to_csv(FEEDBACK_FILE, index=False)
    else:
        feedback_df.to_csv(FEEDBACK_FILE, index=False)

    st.success("✅ Feedback saved to feedback_log.csv!")

    # Re-run the pipeline from training_data onwards
    st.info("🔁 Updating model with new feedback...")

    subprocess.run(["/usr/local/bin/python3.12", "training_data.py"])
    subprocess.run(["/usr/local/bin/python3.12", "train_model.py"])
    subprocess.run(["/usr/local/bin/python3.12", "top3_explanations.py"])

    st.success("✅ Model updated with feedback.")

    st.rerun()



# 7. Optional Reset Button – removes feedback and retrains original model

if st.button("🔄  Reset Feedback and Model"):
    if os.path.exists(FEEDBACK_FILE):
        os.remove(FEEDBACK_FILE)
        st.warning("🗑️ Feedback log reset.")

    st.info("🔄 Rebuilding model with original data...")

    subprocess.run(["/usr/local/bin/python3.12", "data_pipeline.py"])

    st.success("✅ Model and explanations reset.")
    st.rerun()