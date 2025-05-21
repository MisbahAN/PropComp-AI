# This script converts the feature-engineered JSON data into a flat CSV format
# suitable for machine learning, creating positive and negative examples for
# training a ranking model.



# 1. Imports & Constants
#    - json: load JSON data
#    - pandas: data manipulation and CSV output
#    - os: file existence checks
#    - INPUT_FILE / OUTPUT_FILE file paths

import json
import pandas as pd
import os

INPUT_FILE = "../data/engineered/feature_engineered_appraisals_dataset.json"
OUTPUT_FILE = "../data/training/training_data.csv"



# 2. safe_abs(val)
#    - Helper function to safely compute absolute value
#    - Returns None if value is invalid

def safe_abs(val):
    try:
        return abs(val)
    except:
        return None



# 3. build_training_data_from_cleaned(cleaned_file)
#    - Main function that:
#      • Loads feature-engineered JSON
#      • Creates training examples from:
#        - Explicit comps (positive examples)
#        - Candidate pool properties (positive if in comps, negative if not)
#      • Computes absolute differences for all numeric features
#      • Handles duplicate addresses and missing values
#      • Returns pandas DataFrame

def build_training_data_from_cleaned(cleaned_file):
    with open(cleaned_file, "r") as f:
        data = json.load(f)

    rows = []

    for appraisal in data["appraisals"]:
        subject = appraisal["subject"]
        order_id = appraisal.get("orderID", "UNKNOWN")

        # Track comp addresses (lowercased)
        comp_addresses = {
            comp.get("address", "").strip().lower()
            for comp in appraisal.get("comps", [])
        }

        used_addresses = set()

        # Include the comps explicitly
        for comp in appraisal.get("comps", []):
            try:
                address = comp.get("address", "").strip()
                if not address or address.lower() in used_addresses:
                    continue

                row = {
                    "orderID": order_id,
                    "candidate_address": address,
                    "is_comp": 1,

                    "subject_address": subject['address'],

                    "bath_score_diff": comp.get('bath_score_diff'),
                    "full_baths_diff": comp.get('full_baths_diff'),
                    "half_baths_diff": comp.get('half_baths_diff'),
                    "room_count_diff": comp.get('room_count_diff'),
                    "bedrooms_diff": comp.get('bedrooms_diff'),
                    "effective_age_diff": comp.get('effective_age_diff'),
                    "subject_age_diff": comp.get('subject_age_diff'),
                    "lot_size_sf_diff": comp.get('lot_size_diff_sf'),
                    "gla_diff": comp.get('gla_diff'),

                    "abs_bath_score_diff": safe_abs(comp.get("bath_score_diff")),
                    "abs_full_bath_diff": safe_abs(comp.get("full_baths_diff")),
                    "abs_half_bath_diff": safe_abs(comp.get("half_baths_diff")),
                    "abs_room_count_diff": safe_abs(comp.get("room_count_diff")),
                    "abs_bedrooms_diff": safe_abs(comp.get("bedrooms_diff")),
                    "abs_effective_age_diff": safe_abs(comp.get("effective_age_diff")),
                    "abs_subject_age_diff": safe_abs(comp.get("subject_age_diff")),
                    "abs_lot_size_sf_diff": safe_abs(comp.get('lot_size_diff_sf')),
                    "abs_gla_diff": safe_abs(comp.get('gla_diff')),

                    "same_property_type": comp.get("same_property_type", None),
                    "sold_recently": comp.get("sold_recently", None),
                }

                rows.append(row)
                used_addresses.add(address.lower())

            except Exception as e:
                print(f"⚠️ Skipping bad comp in order {order_id}: {e}")
                continue

        # Include properties from candidate pool
        for prop in appraisal.get("properties", []):
            try:
                address = prop.get("address", "").strip()
                if not address or address.lower() in used_addresses:
                    continue

                label = int(address.lower() in comp_addresses)

                row = {
                    "orderID": order_id,
                    "candidate_address": address,
                    "is_comp": label,

                    "subject_address": subject['address'],

                    "bath_score_diff": prop.get('bath_score_diff'),
                    "full_baths_diff": prop.get('full_baths_diff'),
                    "half_baths_diff": prop.get('half_baths_diff'),
                    "room_count_diff": prop.get('room_count_diff'),
                    "bedrooms_diff": prop.get('bedrooms_diff'),
                    "effective_age_diff": prop.get('effective_age_diff'),
                    "subject_age_diff": prop.get('subject_age_diff'),
                    "lot_size_sf_diff": prop.get('lot_size_diff_sf'),
                    "gla_diff": prop.get('gla_diff'),

                    "abs_bath_score_diff": safe_abs(prop.get("bath_score_diff")),
                    "abs_full_bath_diff": safe_abs(prop.get("full_baths_diff")),
                    "abs_half_bath_diff": safe_abs(prop.get("half_baths_diff")),
                    "abs_room_count_diff": safe_abs(prop.get("room_count_diff")),
                    "abs_bedrooms_diff": safe_abs(prop.get("bedrooms_diff")),
                    "abs_effective_age_diff": safe_abs(prop.get("effective_age_diff")),
                    "abs_subject_age_diff": safe_abs(prop.get("subject_age_diff")),
                    "abs_lot_size_sf_diff": safe_abs(prop.get('lot_size_diff_sf')),
                    "abs_gla_diff": safe_abs(prop.get('gla_diff')),

                    "same_property_type": prop.get("same_property_type", None),
                    "sold_recently": prop.get("sold_recently", None),
                }

                rows.append(row)
                used_addresses.add(address.lower())

            except Exception as e:
                print(f"⚠️ Skipping bad property in order {order_id}: {e}")
                continue

    df = pd.DataFrame(rows)
    return df



# 4. Script entry point
#    - Verifies input file exists
#    - Builds training data
#    - Saves to CSV

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f" Input file not found: {INPUT_FILE}")

    df = build_training_data_from_cleaned(INPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved training data to '{OUTPUT_FILE}' with shape: {df.shape}")



# Output: "training_data.csv" containing:
#         - orderID: groups properties by appraisal
#         - candidate_address: property being evaluated
#         - is_comp: 1 if good comp, 0 if not
#         - subject_address: reference property
#         - Various feature differences and their absolute values
#         - same_property_type and sold_recently flags



# =============================================================================
# Example row in training_data.csv:
# {
#   "orderID":            "12345",             # appraisal group ID
#   "candidate_address":  "456 Elm St",        # this property’s address
#   "is_comp":            1,                   # 1=appraiser chose it, 0=didn’t
#   "subject_address":    "123 Main St",       # the reference (subject) property
#
#   # Raw differences (subject minus candidate):
#   "bath_score_diff":    0.5,                 # e.g. 2.5−2.0
#   "full_baths_diff":    1,                   # e.g. 3−2
#   "half_baths_diff":    0,                   # e.g. 1−1
#   "room_count_diff":    2,                   # e.g. 8−6
#   "bedrooms_diff":      1,                   # e.g. 4−3
#   "effective_age_diff": 5,                   # e.g. 20−15
#   "subject_age_diff":   5,                   # e.g. 20−15
#   "lot_size_sf_diff":   200.0,               # e.g. 5000−4800 sqft
#   "gla_diff":           100,                 # e.g. 1500−1400 sqft
#
#   # Absolute values for distance metrics:
#   "abs_bath_score_diff":    0.5,
#   "abs_full_bath_diff":     1,
#   "abs_half_bath_diff":     0,
#   "abs_room_count_diff":    2,
#   "abs_bedrooms_diff":      1,
#   "abs_effective_age_diff": 5,
#   "abs_subject_age_diff":   5,
#   "abs_lot_size_sf_diff":   200.0,
#   "abs_gla_diff":           100,
#
#   # Binary flags from feature_engineering:
#   "same_property_type": 1,  # 1 if this is also a Townhouse/Detached/etc.
#   "sold_recently":       1   # 1 if sold within 90 days of subject date
# }
# =============================================================================