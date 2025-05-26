# This script reads raw appraisal data from a JSON file, applies a series of
# cleaning and normalization steps to key fields, and writes out a cleaned JSON.



# 1. Imports & Constants
#    - json: load and dump JSON
#    - re: regular expressions for parsing
#    - dateutil.parser: flexible date parsing
#    - INPUT_FILE / OUTPUT_FILE file paths

import json
import re
from dateutil import parser

INPUT_FILE = "../data/raw/appraisals_dataset.json"
OUTPUT_FILE = "../data/cleaned/cleaned_appraisals_dataset.json"



# 2. parse_age(val, effective_date)
#    - Normalizes age strings like "new", "built in 1990", "20 years old"
#    - Returns age in years (int) or None

def parse_age(val, effective_date):
    if not val:
        return None

    val = str(val).lower().strip()

    if "new" in val:
        return 0

    match = re.search(r"(\d{1,4})", val)
    if not match:
        return None

    num = int(match.group(1))

    try:
        current_year = parser.parse(effective_date).year
    except:
        return None

    if 999 <= num <= current_year:
        return current_year - num
    else:
        return num



# 3. clean_ages(appraisal)
#    - Applies parse_age to subject, comps, and properties

def clean_ages(appraisal):
    subject = appraisal['subject']

    subject['subject_age'] = parse_age(subject.get('subject_age'), subject.get('effective_date'))
    subject['effective_age'] = parse_age(subject.get('effective_age'), subject.get('effective_date'))

    for comp in appraisal['comps']:
        comp['age'] = parse_age(comp.get('age'), comp.get('sale_date'))

    for property in appraisal['properties']:
        property['age'] = parse_age(property.get('year_built'), property.get('close_date'))

    return appraisal



# 4. parse_gla(val)
#    - Extracts and standardizes gross living area

def parse_gla(val):
    if not val:
        return None

    val = str(val).lower().replace(',', '').strip()
    tokens = val.split()

    match = re.search(r"(\d+(?:\.\d+)?)", val)
    if not match:
        return None

    number = float(match.group(1))

    if "sqm" in tokens or "sq m" in tokens:
        number *= 10.7639

    return int(round(number))



# 5. clean_glas(appraisal)
#    - Applies parse_gla to subject, comps, and properties

def clean_glas(appraisal): 
    subject = appraisal['subject']
    subject['gla'] = parse_gla(subject.get('gla'))

    for comp in appraisal['comps']:
        comp['gla'] = parse_gla(comp.get('gla'))

    for property in appraisal['properties']:
        property['gla'] = parse_gla(property.get('gla'))

    return appraisal



# 6. parse_lot_size(val)
#    - Parses lot size and converts units to sqft

def parse_lot_size(val):
    if not val:
        return None

    original_val = str(val).lower().replace(",", "").strip()
    val = original_val

    if "n/a" in val or "condo" in val or "common" in val or val in {"sqft", "sqm", ""}:
        return None

    if "/" in val:
        val = val.split("/")[-1].strip()

    val = re.sub(r"(sf|sqft|sqm|acres?|\+/-|±|m|ft|')", "", val).strip()

    match = re.search(r"(\d+(?:\.\d+)?)", val)
    if not match:
        return None

    number = float(match.group(1))

    if "sqm" in original_val:
        number *= 10.7639
    elif "acre" in original_val or "ac" in original_val:
        number *= 43560

    return float(round(number))



# 7. clean_lot_sizes(appraisal)
#    - Applies parse_lot_size to subject, comps, and properties

def clean_lot_sizes(appraisal):
    subject = appraisal['subject']
    subject['lot_size_sf'] = parse_lot_size(subject.get('lot_size_sf'))

    for comp in appraisal['comps']:
        comp['lot_size_sf'] = parse_lot_size(comp.get('lot_size'))

    for property in appraisal['properties']:
        property['lot_size_sf'] = parse_lot_size(property.get('lot_size_sf'))

    return appraisal



# 8. parse_total_rooms(val)
#    - Parses "3+1" style values or plain ints for total rooms

def parse_total_rooms(val):
    if not val:
        return None
    if "+" in str(val):
        nums = val.split('+')
        return int(nums[0]) + int(nums[1])
    return int(val)



# 9. clean_total_rooms(appraisal)
#    - Applies parse_total_rooms to room_count field

def clean_total_rooms(appraisal):
    subject = appraisal['subject']
    subject['room_count'] = parse_total_rooms(subject.get('room_count'))

    for comp in appraisal['comps']:
        comp['room_count'] = parse_total_rooms(comp.get('room_count'))

    for property in appraisal['properties']:
        property['room_count'] = parse_total_rooms(property.get('room_count'))

    return appraisal



# 10. clean_bedrooms(appraisal)
#     - Applies parse_total_rooms to bedroom fields

def clean_bedrooms(appraisal):
    subject = appraisal['subject']
    subject['num_beds'] = parse_total_rooms(subject.get('num_beds'))

    for comp in appraisal['comps']:
        comp['num_beds'] = parse_total_rooms(comp.get('bed_count'))

    for property in appraisal['properties']:
        property['num_beds'] = parse_total_rooms(property.get('bedrooms'))

    return appraisal



# 11. get_bath_score(val=None, full=None, half=None)
#     - Calculates bath score = full + 0.5 * half


def get_bath_score(val=None, full=None, half=None):
    try:
        if val:
            val = str(val).strip().upper()
            f_match = re.search(r"(\d+)\s*F", val)
            h_match = re.search(r"(\d+)\s*H", val)

            if f_match or h_match:
                full = int(f_match.group(1)) if f_match else 0
                half = int(h_match.group(1)) if h_match else 0
            elif ":" in val:
                nums = val.split(":")
                full = int(nums[0])
                half = int(nums[1])
            elif val.isdigit():
                full = int(val)
                half = 0
            else:
                return None, 0, 0 
        else:
            full = int(float(full or 0))
            half = int(float(half or 0))

        score = full + 0.5 * half
        return score, full, half

    except:
        return None, 0, 0



# 12. clean_baths(appraisal)
#     - Applies get_bath_score to various bath formats

def clean_baths(appraisal):
    subject = appraisal['subject']
    score, full, half = get_bath_score(val=subject.get('num_baths'))
    subject['bath_score'] = score
    subject['num_full_baths'] = full
    subject['num_half_baths'] = half

    for comp in appraisal['comps']:
        score, full, half = get_bath_score(val=comp.get('bath_count'))
        comp['bath_score'] = score
        comp['num_full_baths'] = full
        comp['num_half_baths'] = half

    for property in appraisal['properties']:
        score, full, half = get_bath_score(full=property.get('full_baths'), half=property.get('half_baths'))
        property['bath_score'] = score
        property['num_full_baths'] = full
        property['num_half_baths'] = half

    return appraisal



# 13. clean_conditions(appraisal)
#     - Tracks unique condition values in global lists

unique_subject_conditions = []
unique_comp_conditions = []
unique_property_conditions = []

def clean_conditions(appraisal):
    subject = appraisal['subject']
    subject_cond = subject.get('condition')
    if subject_cond not in unique_subject_conditions:
        unique_subject_conditions.append(subject_cond)

    for comp in appraisal['comps']:
        comp_cond = comp.get('condition')
        if comp_cond not in unique_comp_conditions:
            unique_comp_conditions.append(comp_cond)



# 14. parse_comp_dist(val)
#     - Parses comp distance in km from string

def parse_comp_dist(val):
    if not val or not isinstance(val, str):
        return None
    try:
        return float(val.strip().lower().replace("km", "").strip())
    except ValueError:
        return None



# 15. clean_comp_distances(appraisal)
#     - Applies parse_comp_dist to each comp

def clean_comp_distances(appraisal):
    for comp in appraisal.get('comps'):
        comp['distance_to_subject_km'] = parse_comp_dist(comp.get('distance_to_subject'))
    return appraisal



# 16. safe_float(val)
#     - Safely parses a float-like value to int, if possible

def safe_float(val):
    try:
        return int(str(val).replace(",", "").strip())
    except:
        return None



# 17. clean_sale_price(appraisal)
#     - Standardizes sale price on comps and properties

def clean_sale_price(appraisal):
    for comp in appraisal.get("comps"):
        comp['sale_price'] = safe_float(comp.get('sale_price'))

    for property in appraisal.get("properties"):
        property['sale_price'] = safe_float(property.get('close_price'))

    return appraisal



# 18. clean_all_data()
#     - Main orchestration function that applies all cleaners and saves output

def clean_all_data():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    cleaned = []
    for appraisal in data["appraisals"]:
        clean_ages(appraisal)
        clean_glas(appraisal)
        clean_lot_sizes(appraisal)
        clean_total_rooms(appraisal)
        clean_bedrooms(appraisal)
        clean_baths(appraisal)
        clean_conditions(appraisal)
        clean_sale_price(appraisal)
        clean_comp_distances(appraisal)
        cleaned.append(appraisal)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"appraisals": cleaned}, f, indent=2)

    print(f"Saved cleaned JSON to {OUTPUT_FILE}")



if __name__ == "__main__":
    clean_all_data()