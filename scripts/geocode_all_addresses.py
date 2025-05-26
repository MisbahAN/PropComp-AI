# This script geocodes a list of missing addresses using Nominatim and GPT-assisted cleanup.
# It attempts to resolve each address to lat/lon, falling back to GPT for cleaning if needed.
# Caches successful results in a JSON file to avoid repeated geocoding calls.


# 1. Imports & Config
#    - os/json/time: file and sleep handling
#    - tqdm: progress bar
#    - geopy: geocoder and timeout handling
#    - openai: fallback address cleaning

import os
import json
import time
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "../data/geocoded-data/geocoded_addresses.json"
MISSING_FILE = "../data/geocoded-data/missing_addresses.txt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)



# 2. normalize_address(address)
#    - Normalizes an address string by lowering case and trimming spaces

def normalize_address(address):
    return address.lower().strip()



# 3. safe_geocode(geolocator, address)
#    - Attempts to geocode an address using Nominatim with retry on timeout

def safe_geocode(geolocator, address):
    try:
        return geolocator.geocode(address, timeout=10)
    except GeocoderTimedOut:
        time.sleep(2)
        return safe_geocode(geolocator, address)
    except Exception as e:
        print(f"Geocode error for '{address}': {e}")
        return None



# 4. clean_address_with_gpt(raw_address)
#    - Uses GPT to clean and reformat a raw address into Canadian mailing format

def clean_address_with_gpt(raw_address):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a geocoding assistant trained to clean and standardize Canadian mailing addresses strictly "
                    "for geolocation purposes.\n\n"
                    "Your job is to rewrite any input into this format exactly:\n"
                    "[unit-civic number] [Street Name Capitalized], [City Capitalized], [Province Abbreviation] [Postal Code], Canada\n\n"
                    "Rules:\n"
                    "- Use commas between address parts (street, city, province, postal code, country)\n"
                    "- Ensure proper capitalization (e.g., 'Kemptville', 'ON')\n"
                    "- Postal codes must have a space between the 3rd and 4th character (e.g., 'T2N 3B8')\n"
                    "- If the address includes a unit/civic format (e.g., '119 110'), rewrite it as '110-119'\n"
                    "- Do not include neighborhood names, regions, or repetitions — just the precise mailing address\n"
                    "- Your response must only include the final cleaned address, with no explanation or extra text"
                )},
                {"role": "user", "content": f"Please clean and standardize this address: {raw_address}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error for '{raw_address}': {e}")
        return None



# 5. Load Cache and Missing Address List
#    - Loads previously geocoded results and addresses still requiring lookup

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        geocoded = json.load(f)
else:
    geocoded = {}

with open(MISSING_FILE, "r") as f:
    missing_addresses = [normalize_address(line) for line in f if line.strip()]



# 6. Geocode Missing Addresses
#    - Attempts Nominatim first, then GPT cleanup fallback

geolocator = Nominatim(user_agent="comp-geocoder")
added = 0

for raw_address in tqdm(missing_addresses):
    if raw_address in geocoded and geocoded[raw_address] is not None:
        continue

    print(f"\U0001F4CD Geocoding: {raw_address}")
    location = safe_geocode(geolocator, raw_address)
    if location:
        geocoded[raw_address] = {
            "lat": location.latitude,
            "lon": location.longitude,
        }
        added += 1
    else:
        print(f"⚠️ Nominatim failed. Trying GPT to clean: {raw_address}")
        cleaned = clean_address_with_gpt(raw_address)
        if cleaned:
            location = safe_geocode(geolocator, cleaned)
            if location:
                print(f"GPT cleaned success: {cleaned}")
                geocoded[raw_address] = {
                    "lat": location.latitude,
                    "lon": location.longitude,
                }
                added += 1
            else:
                print(f"GPT cleaned address failed to geocode: {cleaned}")
                geocoded[raw_address] = None
        else:
            print(f"GPT failed to parse: {raw_address}")
            geocoded[raw_address] = None

    # Save incrementally
    with open(CACHE_FILE, "w") as f:
        json.dump(geocoded, f, indent=2)

    time.sleep(1)



# 7. Completion Message
#    - Final output to terminal after geocoding completes

print(f"\nGeocoding complete — {added} new addresses added to {CACHE_FILE}")