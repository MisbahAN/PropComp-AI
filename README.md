# Property Recommendation System

A machine learning system that identifies and explains the best comparable properties for real estate appraisals. Built during the first sprint of Headstarter SWE Residency.

## Project Overview

This system processes property appraisal data through several stages:
1. Data cleaning and standardization
2. Feature engineering
3. Training a ranking model
4. Generating human-readable explanations for recommendations

## Folder Structure

```
📁 property-recommendation-system/
│
├── 📁 data/
│   ├── raw/              # Original appraisal data
│   ├── cleaned/          # Standardized data
│   ├── engineered/       # Feature-enhanced data
│   ├── training/         # ML-ready data
│   └── README.md         # Data documentation
│
├── 📁 models/            # Trained ML models
├── 📁 scripts/           # Python processing scripts
├── 📁 outputs/           # Generated explanations
├── 📁 notebooks/         # Optional analysis notebooks
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Scripts Overview

- `clean_initial_data.py`: Standardizes property data (ages, sizes, rooms, etc.)
- `features.py`: Adds engineered features (property type matching, time-based flags)
- `training_data.py`: Converts data to ML format with positive/negative examples
- `train_model.py`: Trains XGBoost ranking model
- `top3_explanations.py`: Generates human-readable explanations using SHAP and GPT

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

1. Ensure you have the raw data file in `data/raw/appraisals_dataset.json`
2. Run the scripts in order:
   ```bash
   cd scripts
   python clean_initial_data.py
   python features.py
   python training_data.py
   python train_model.py
   python top3_explanations.py
   ```

## Requirements

- Python 3.8+
- OpenAI API key (for explanations)
- See `requirements.txt` for full dependency list

## Author

Misbah Ahmed Nauman
- Portfolio: [misbahan.com](https://misbahan.com)
- Built during Headstarter SWE Residency Sprint 1 