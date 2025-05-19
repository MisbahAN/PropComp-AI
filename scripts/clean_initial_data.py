# This script reads raw appraisal data from a JSON file, applies a series of
# cleaning and normalization steps to key fields, and writes out a cleaned JSON.
#
# Steps (in order):
#
# 1. Imports & Constants
#    - json: load and dump JSON
#    - re: regular expressions for parsing
#    - dateutil.parser: flexible date parsing
#    - INPUT_FILE / OUTPUT_FILE file paths
#
# 2. parse_age(val, effective_date)
#    - Normalizes age strings like "new", "built in 1990", "20 years old"
#    - Returns age in years (int) or None
#    - If “new” → 0; if numeric looks like a year → current_year – year; else returns number
#
# 3. clean_ages(appraisal)
#    - Applies parse_age to:
#        • Subject property’s subject_age & effective_age (using its effective_date)
#        • Each comp’s age (using sale_date)
#        • Each property’s age (using close_date)
#
# 4. parse_gla(val)
#    - Extracts numeric GLA (gross living area) value
#    - Converts “sqm” or “sq m” → square feet (×10.7639)
#    - Rounds and returns integer sqft or None
#
# 5. clean_glas(appraisal)
#    - Runs parse_gla on the subject, each comp, and each property
#
# 6. parse_lot_size(val)
#    - Cleans lot size strings (e.g., “0.2 acres”, “600 sqm”, “20×30 / 600 sqft”)
#    - Strips unit markers, handles acres → sqft (×43 560) and sqm → sqft (×10.7639)
#    - Rounds and returns float sqft or None
#
# 7. clean_lot_sizes(appraisal)
#    - Applies parse_lot_size to subject.lot_size_sf, each comp.lot_size, and
#      each property.lot_size_sf
#
# 8. parse_total_rooms(val)
#    - Parses room counts, handling “3+1” by summing, else int(val)
#
# 9. clean_total_rooms(appraisal)
#    - Cleans overall room_count on subject, comps, and properties
#
# 10. clean_bedrooms(appraisal)
#    - Cleans bedroom counts (num_beds / bed_count / bedrooms) via parse_total_rooms
#
# 11. get_bath_score(val=None, full=None, half=None)
#    - Computes bathroom score = full + 0.5 × half
#    - Supports “2:1” format or separate full_baths, half_baths args
#    - Returns (score, full_int, half_int)
#
# 12. clean_baths(appraisal)
#    - For subject and comps: reads num_baths or bath_count, applies get_bath_score
#    - For properties: reads full_baths & half_baths, applies get_bath_score
#    - Sets bath_score, num_full_baths, num_half_baths on each record
#
# 13. clean_conditions(appraisal)
#    - Collects unique condition strings from subject, comps, properties into lists
#      for later analysis or reporting
#
# 14. clean_all_data()
#    - Loads raw data from INPUT_FILE
#    - Iterates each appraisal and applies all clean_* functions in sequence
#    - Prints collected unique condition values
#    - Writes cleaned data to OUTPUT_FILE (indented JSON)
#
# 15. Script entry point
#    if __name__ == "__main__":
#        clean_all_data()
#
# Output: “cleaned_appraisals_dataset.json” containing standardized numeric
#         fields ready for downstream modeling/comparison.

import json
import re
from dateutil import parser

INPUT_FILE = "../data/raw/appraisals_dataset.json"
OUTPUT_FILE = "../data/cleaned/cleaned_appraisals_dataset.json"

def parse_age(val, effective_date):
    if not val:
        return None

    val = str(val).lower().strip()

    if "new" in val:
        return 0

    # Extract number (1 to 4 digits)
    match = re.search(r"(\d{1,4})", val)
    if not match:
        return None

    num = int(match.group(1))

    # Get year from effective date
    try:
        current_year = parser.parse(effective_date).year
    except:
        return None

    if 999 <= num <= current_year:
        return current_year - num
    else:
        return num

def clean_ages(appraisal):
    subject = appraisal['subject']

    subject_age = subject.get('subject_age')
    subject_effective_age = subject.get('effective_age')
    effective_date = subject.get('effective_date')
    
    subject['subject_age'] = parse_age(subject_age, effective_date)
    subject['effective_age'] = parse_age(subject_effective_age, effective_date)

    for comp in appraisal['comps']:
        comp_age = comp.get('age')
        comp_sale_date = comp.get('sale_date')
        comp['age'] = parse_age(comp_age, comp_sale_date)

    for property in appraisal['properties']:
        year_built = property.get('year_built')
        close_date = property.get('close_date')
        property['age'] = parse_age(year_built, close_date)

    return appraisal

def parse_gla(val):
    if not val:
        return None

    # print(val)
    # print(type(val))   

    val = str(val).lower().replace(',', '').strip()
    tokens = val.split()

    match = re.search(r"(\d+(?:\.\d+)?)", val)
    if not match:
        return None

    number = float(match.group(1))

    if "sqm" in tokens or "sq m" in tokens:
        number *= 10.7639

    return int(round(number))

def clean_glas(appraisal): 

    subject = appraisal['subject']

    subject_gla = subject.get('gla')
    subject['gla'] = parse_gla(subject_gla)

    for comp in appraisal['comps']:
        comp_gla = comp.get('gla')
        comp['gla'] = parse_gla(comp_gla)
        
    for property in appraisal['properties']:
        property_gla = property.get('gla')
        property['gla'] = parse_gla(property_gla)

    return appraisal

def parse_lot_size(val):
    if not val:
        return None

    original_val = str(val).lower().replace(",", "").strip()
    val = original_val

    if "n/a" in val or "condo" in val or "common" in val or val in {"sqft", "sqm", ""}:
        return None

    # Use the RHS of a slash if present (e.g., dimensions / area)
    if "/" in val:
        val = val.split("/")[-1].strip()

    # Remove trailing junk so regex works
    val = re.sub(r"(sf|sqft|sqm|acres?|\+/-|±|m|ft|')", "", val).strip()

    # Extract the first number
    match = re.search(r"(\d+(?:\.\d+)?)", val)
    if not match:
        return None

    number = float(match.group(1))

    # Check unit in original value 
    if "sqm" in original_val:
        number *= 10.7639
    elif "acre" in original_val or "ac" in original_val:
        number *= 43560

    # Otherwise assume sqft

    return float(round(number))

def clean_lot_sizes(appraisal):

    subject = appraisal['subject']

    subject_lot_size = subject.get('lot_size_sf')
    subject['lot_size_sf'] = parse_lot_size(subject_lot_size)

    for comp in appraisal['comps']:
        comp_lot_size = comp.get('lot_size')
        comp['lot_size_sf'] = parse_lot_size(comp_lot_size)
    
        
    for property in appraisal['properties']:
        property_lot_size = property.get('lot_size_sf')
        property['lot_size_sf'] = parse_lot_size(property_lot_size)

    return appraisal

def parse_total_rooms(val):

    if not val:
        return None
    
    if "+" in str(val):
        nums = val.split('+')
        return int(nums[0]) + int(nums[1])

    return int(val)

def clean_total_rooms(appraisal):
    subject = appraisal['subject']

    subject_rooms = subject.get('room_count')
    subject['room_count'] = parse_total_rooms(subject_rooms)

    for comp in appraisal['comps']:
        comp_rooms = comp.get('room_count')
        comp['room_count'] = parse_total_rooms(comp_rooms)

    for property in appraisal['properties']:
        property_rooms = property.get('room_count')
        property['room_count'] = parse_total_rooms(property_rooms)
        
    return appraisal

def clean_bedrooms(appraisal):
    subject = appraisal['subject']

    subject_bedrooms = subject.get('num_beds')
    subject['num_beds'] = parse_total_rooms(subject_bedrooms)

    for comp in appraisal['comps']:
        comp_bedrooms = comp.get('bed_count')
        comp['bed_count'] = parse_total_rooms(comp_bedrooms)

    for property in appraisal['properties']:
        property_bedrooms = property.get('bedrooms')
        property['bedrooms'] = parse_total_rooms(property_bedrooms)
        
    return appraisal

def get_bath_score(val=None, full=None, half=None):
    try:
        if val:
            nums = val.strip().split(':')
            full = float(nums[0])
            half = float(nums[1])
        else:
            full = float(full or 0)
            half = float(half or 0)

        score = full + 0.5 * half
        return score, int(full), int(half)

    except:
        return None, 0, 0

def clean_baths(appraisal):
    subject = appraisal['subject']
    subject_baths = subject.get('num_baths')  
    score, full, half = get_bath_score(val=subject_baths)
    subject['bath_score'] = score
    subject['num_full_baths'] = full
    subject['num_half_baths'] = half

    for comp in appraisal['comps']:
        comp_baths = comp.get('bath_count')
        score, full, half = get_bath_score(val=comp_baths)
        comp['bath_score'] = score
        comp['num_full_baths'] = full
        comp['num_half_baths'] = half

    for property in appraisal['properties']:
        full = property.get('full_baths')
        half = property.get('half_baths')
        score, full, half = get_bath_score(full=full, half=half)
        property['bath_score'] = score
        property['num_full_baths'] = full
        property['num_half_baths'] = half

    return appraisal


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

       

        cleaned.append(appraisal)

    print(unique_subject_conditions)
    print(unique_comp_conditions)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"appraisals": cleaned}, f, indent=2)

    print(f"Saved cleaned JSON to {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_all_data()
