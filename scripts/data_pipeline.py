# This script coordinates the end-to-end preprocessing and modeling pipeline for real estate appraisals.
# It conditionally runs address geocoding based on cache status and always reruns downstream steps.



# 1. Imports & Utilities
#    - os, subprocess: for running shell commands and checking file presence
#    - json: to load and inspect cached and required addresses

import os
import subprocess
import json
import sys



# 2. normalize_address(address)
#    - Normalizes an address string to a consistent lowercase format with abbreviated street types

def normalize_address(address):
    if not address:
        return None
    return address.lower().strip()\
        .replace("drive", "dr")\
        .replace("road", "rd")\
        .replace("street", "st")\
        .replace("avenue", "ave")



# 3. should_run_geocoding()
#    - Checks if geocoding script needs to run by comparing addresses in dataset vs cached geocodes

def should_run_geocoding():
    cache_path = "../data/geocoded-data/geocoded_addresses.json"
    data_path = "../data/cleaned/cleaned_appraisals_dataset.json"

    if not os.path.exists(cache_path):
        return True

    with open(cache_path, "r") as f:
        cached = set(json.load(f).keys())

    with open(data_path, "r") as f:
        data = json.load(f)

    needed = set()
    for appraisal in data.get("appraisals", []):
        all_addresses = (
            [appraisal.get("subject", {}).get("address", "")]
            + [comp.get("address", "") for comp in appraisal.get("comps", [])]
            + [prop.get("address", "") for prop in appraisal.get("properties", [])]
        )

        for raw_addr in all_addresses:
            norm = normalize_address(raw_addr)
            if norm:
                needed.add(norm)

    missing = [addr for addr in needed if addr not in cached]

    return len(missing) > 0



# 4. run(script)
#    - Utility function to run a Python script with subprocess and print status

def run(script):
    print(f"\nRunning {script} ...")
    subprocess.run([sys.executable, script], check=True)



# 5. Main Execution Pipeline
#    - Stage 1: Clean raw data
#    - Stage 2: Conditionally run geocoder
#    - Stage 3–6: Always run feature creation, training data prep, model training, and explanations

run("clean_initial_data.py")

if should_run_geocoding():
    run("geocode_all_addresses.py")
else:
    print("All addresses already geocoded — skipping.")

run("features.py")
run("training_data.py")
run("train_model.py")
run("top3_explanations.py")