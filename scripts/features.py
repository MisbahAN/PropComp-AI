# This script takes cleaned appraisal data and adds engineered features to improve
# the model's ability to identify good comparable properties.
#
# Steps (in order):
#
# 1. Imports & Constants
#    - json: load and dump JSON
#    - dateutil.parser: flexible date parsing
#    - fuzzywuzzy.process: fuzzy string matching for property types
#    - INPUT_FILE / OUTPUT_FILE file paths
#    - CANONICAL_TYPES: standardized property type list
#    - manual_type_map: mapping of variations to canonical types
#
# 2. sold_recently(appraisal)
#    - Adds binary flag (1/0) if property sold within 90 days of subject's effective date
#    - Uses dateutil.parser for flexible date parsing
#
# 3. map_to_property_type(raw)
#    - Normalizes property type strings to canonical types
#    - First tries exact match in manual_type_map
#    - Falls back to fuzzy matching if no exact match
#
# 4. same_property_type(appraisal)
#    - Adds binary flag (1/0) if comp/property matches subject's type
#    - Uses map_to_property_type for normalization
#
# 5. effective_age_diff(appraisal)
#    - Computes age difference between subject and comps/properties
#    - Uses effective_age for subject and age for others
#
# 6. subject_age_diff(appraisal)
#    - Similar to effective_age_diff but uses subject_age
#
# 7. lot_size_diff(appraisal)
#    - Computes lot size difference in square feet
#
# 8. gla_diff(appraisal)
#    - Computes gross living area difference in square feet
#
# 9. room_diff(appraisal)
#    - Computes total room count difference
#
# 10. bedroom_diff(appraisal)
#    - Computes bedroom count difference
#
# 11. bath_score_diff(appraisal)
#    - Computes bathroom score difference (full + 0.5*half)
#
# 12. full_bath_diff(appraisal)
#    - Computes full bathroom count difference
#
# 13. half_bath_diff(appraisal)
#    - Computes half bathroom count difference
#
# 14. add_new_features()
#    - Main function that:
#      • Loads cleaned data
#      • Applies all feature engineering functions
#      • Saves enhanced dataset
#
# Output: "feature_engineered_appraisals_dataset.json" containing original data
#         plus new engineered features for modeling

import json
from dateutil import parser
from fuzzywuzzy import process

INPUT_FILE = "../data/cleaned/cleaned_appraisals_dataset.json"
OUTPUT_FILE = "../data/engineered/feature_engineered_appraisals_dataset.json"

CANONICAL_TYPES = [
    "Townhouse", "Detached", "Condominium", "Semi Detached",
    "High Rise Apartment", "Low Rise Apartment", "Duplex", "Triplex", "Fourplex"
]

manual_type_map = {
    "rural resid": "Detached",
    "rural residential": "Detached",
    "single family": "Detached",
    "single family residence": "Detached",
    "overunder": "Duplex",
    "4 plex": "Fourplex",
    "triplex": "Triplex",
    "duplex": "Duplex",
    "over under": "Duplex",
    "condo apt": "Condominium",
    "condo apartment": "Condominium",
    "condo/apt unit": "Condominium",
    "common element condo": "Condominium",
    "row unit": "Townhouse",
    "row unit 2 storey": "Townhouse",
    "row unit 3 storey": "Townhouse",
    "stacked": "Townhouse",
    "mobiletrailer": "Detached",
    "mobile home": "Detached",
    "mobile": "Detached",
    "link": "Semi Detached",
    "farm": "Detached",
    "vacant land": None,
    "residential land": None,
    "residential": "",
    "locker": None,
    "other": None,
    "": None,
    None: None
}


def sold_recently(appraisal):
    subject = appraisal['subject']
    subject_effective_data = parser.parse(subject['effective_date'])

    for comp in appraisal['comps']:
        sale_date = parser.parse(comp.get('sale_date'))
        days_ago_sold = (subject_effective_data-sale_date).days
        if days_ago_sold <= 90:
            comp['sold_recently'] = 1
        else:
            comp['sold_recently'] = 0
        

    for property in appraisal['properties']:
        close_date = parser.parse(property.get('close_date'))
        days_ago_sold = (subject_effective_data-close_date).days
        if days_ago_sold <= 90:
            property['sold_recently'] = 1
        else:
            property['sold_recently'] = 0

    return appraisal

def map_to_property_type(raw):
    if not raw:
        return None

    val = str(raw).lower().strip().replace(",", "").replace("-", " ")

    # 1. Manual check first
    if val in manual_type_map:
        return manual_type_map[val]

    # 2. Fuzzy fallback to catch close things like "semi detached"
    match, score = process.extractOne(val, CANONICAL_TYPES, scorer=process.fuzz.partial_ratio)
    return match if score >= 80 else None


def same_property_type(appraisal):
    subject = appraisal['subject']
    subject_raw = subject.get('structure_type')
    subject_type = map_to_property_type(subject_raw)

    if not subject_type:
        return appraisal  # skip if undefined/none

    for comp in appraisal['comps']:
        comp_raw = comp.get('prop_type')
        comp_type = map_to_property_type(comp_raw)
        comp['same_property_type'] = int(subject_type == comp_type)

    for property in appraisal['properties']:
        property_raw = property.get('property_sub_type')
        property_type = map_to_property_type(property_raw)
        property['same_property_type'] = int(subject_type == property_type)

    return appraisal

def effective_age_diff(appraisal):
    subject = appraisal['subject']
    subject_effective_age = subject.get("effective_age")
    
    if not subject_effective_age:
        return appraisal

    for comp in appraisal['comps']:
        comp_age = comp.get('age')
        if comp_age:
            comp['effective_age_diff'] = subject_effective_age-comp_age
        else:
            comp['effective_age_diff'] = None

    for property in appraisal['properties']:
        property_age = property.get('age')
        if property_age:
            property['effective_age_diff'] = subject_effective_age-property_age
        else:
            property['effective_age_diff'] = None

    return appraisal

def subject_age_diff(appraisal):
    subject = appraisal['subject']
    subject_age = subject.get("subject_age")

    if not subject_age:
        return appraisal
    
    for comp in appraisal['comps']:
        comp_age = comp.get('age')
        if comp_age:
            comp['subject_age_diff'] = subject_age-comp_age
        else:
            comp['subject_age_diff'] = None

    for property in appraisal['properties']:
        property_age = property.get('age')
        if property_age:
            property['subject_age_diff'] = subject_age-property_age
        else:
            property['subject_age_diff'] = None

    return appraisal

def lot_size_diff(appraisal):
    subject = appraisal['subject']
    subject_lot_size = subject.get('lot_size_sf')

    for comp in appraisal['comps']:
        comp_lot_size = comp.get('lot_size_sf')
        if subject_lot_size is not None and comp_lot_size is not None:
            comp['lot_size_diff_sf'] = subject_lot_size - comp_lot_size
        else:
            comp['lot_size_diff_sf'] = None

    for property in appraisal['properties']:
        property_lot_size = property.get('lot_size_sf')
        if subject_lot_size is not None and property_lot_size is not None:
            property['lot_size_diff_sf'] = subject_lot_size - property_lot_size
        else:
            property['lot_size_diff_sf'] = None

    return appraisal

def gla_diff(appraisal):
    subject = appraisal['subject']
    subject_gla = subject.get('gla')

    if not subject_gla:
        return appraisal
    
    for comp in appraisal['comps']:
        comp_gla = comp.get('gla')
        if comp_gla:
            comp['gla_diff'] = subject_gla - comp_gla
        else:
            comp['gla_diff'] = None

    for property in appraisal['properties']:
        property_gla = property.get('gla')
        if property_gla:
            property['gla_diff'] = subject_gla - property_gla
        else:
            property['gla_diff'] = None

    return appraisal

def room_diff(appraisal):
    subject = appraisal['subject']
    subject_rooms = subject.get('room_count')

    if not subject_rooms:
        return appraisal

    for comp in appraisal['comps']:
        comp_rooms = comp.get('room_count')
        if comp_rooms:
            comp['room_count_diff'] = subject_rooms - comp_rooms
        else:
            comp['room_count_diff'] = None

    for property in appraisal['properties']:
        property_rooms = property.get('room_count')
        if property_rooms:
            property['room_count_diff'] = subject_rooms - property_rooms
        else:
            property['room_count_diff'] = None
    
    return appraisal

def bedroom_diff(appraisal):
    subject = appraisal['subject']
    subject_bedrooms = subject.get('num_beds')

    if not subject_bedrooms:
        return appraisal

    for comp in appraisal['comps']:
        comp_bedrooms = comp.get('bed_count')
        if comp_bedrooms:
            comp['bedrooms_diff'] = subject_bedrooms - comp_bedrooms
        else:
            comp['bedrooms_diff'] = None

    for property in appraisal['properties']:
        property_bedrooms = property.get('bedrooms')
        if property_bedrooms:
            property['bedrooms_diff'] = subject_bedrooms - property_bedrooms
        else:
            property['bedrooms_diff'] = None
    
    return appraisal

def bath_score_diff(appraisal):
    subject = appraisal['subject']
    subject_bath_score = subject.get('bath_score')

    if not subject_bath_score:
        return appraisal

    for comp in appraisal['comps']:
        comp_bath_score = comp.get('bath_score')
        if comp_bath_score:
            comp['bath_score_diff'] = subject_bath_score - comp_bath_score
        else:
            comp['bath_score_diff'] = None

    for property in appraisal['properties']:
        property_bath_score = property.get('bath_score')
        if property_bath_score:
            property['bath_score_diff'] = subject_bath_score - property_bath_score
        else:
            property['bath_score_diff'] = None
    
    return appraisal

def full_bath_diff(appraisal):
    subject = appraisal['subject']
    subject_fulls = subject.get('num_full_baths')

    if not subject_fulls:
        return appraisal

    for comp in appraisal['comps']:
        comp_fulls = comp.get('num_full_baths')
        if comp_fulls:
            comp['full_baths_diff'] = subject_fulls - comp_fulls
        else:
            comp['full_baths_diff'] = None

    for property in appraisal['properties']:
        property_fulls = property.get('num_full_baths')
        if property_fulls:
            property['full_baths_diff'] = subject_fulls - property_fulls
        else:
            property['full_baths_diff'] = None
    
    return appraisal

def half_bath_diff(appraisal):
    subject = appraisal['subject']
    subject_halfs = subject.get('num_half_baths')

    if not subject_halfs:
        return appraisal

    for comp in appraisal['comps']:
        comp_halfs = comp.get('num_half_baths')
        if comp_halfs:
            comp['half_baths_diff'] = subject_halfs - comp_halfs
        else:
            comp['half_baths_diff'] = None

    for property in appraisal['properties']:
        property_halfs = property.get('num_half_baths')
        if property_halfs:
            property['half_baths_diff'] = subject_halfs - property_halfs
        else:
            property['half_baths_diff'] = None
    
    return appraisal

def add_new_features():
    with open(INPUT_FILE, "r") as f:
            data = json.load(f)
    
    feature_engineered = []
    
    for appraisal in data["appraisals"]:
        
        sold_recently(appraisal)
        same_property_type(appraisal)
        effective_age_diff(appraisal)
        subject_age_diff(appraisal)
        lot_size_diff(appraisal)
        gla_diff(appraisal)
        room_diff(appraisal)
        bedroom_diff(appraisal)
        bath_score_diff(appraisal)
        full_bath_diff(appraisal)
        half_bath_diff(appraisal)

        feature_engineered.append(appraisal)


    with open(OUTPUT_FILE, "w") as f:
        json.dump({"appraisals": feature_engineered}, f, indent=2)

    print(f"Saved cleaned JSON to {OUTPUT_FILE}")
    

if __name__ == "__main__":
    add_new_features()    

