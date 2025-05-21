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
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Scripts Overview

### Data Processing Pipeline

1. `clean_initial_data.py` (375 lines)
   - Standardizes property data
   - Handles missing values
   - Normalizes property features (ages, sizes, rooms)
   - Outputs cleaned data to `data/cleaned/`

2. `features.py` (403 lines)
   - Adds engineered features
   - Implements property type matching
   - Creates time-based flags
   - Outputs enhanced data to `data/engineered/`

3. `training_data.py` (173 lines)
   - Converts data to ML format
   - Creates positive/negative examples
   - Prepares data for model training
   - Outputs to `data/training/`

4. `train_model.py` (116 lines)
   - Trains XGBoost ranking model
   - Implements cross-validation
   - Saves model to `models/`

5. `top3_explanations.py` (164 lines)
   - Generates human-readable explanations
   - Uses SHAP for feature importance
   - Integrates with GPT for natural language explanations
   - Outputs to `outputs/`

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
- Dependencies (see requirements.txt):
  - pandas >= 1.3.0
  - xgboost >= 1.5.0
  - scikit-learn >= 0.24.0
  - numpy >= 1.19.0
  - fuzzywuzzy >= 0.18.0
  - python-Levenshtein >= 0.12.0
  - shap >= 0.41.0
  - openai >= 1.0.0
  - python-dotenv >= 0.19.0

## Author

Misbah Ahmed Nauman
- Portfolio: [misbahan.com](https://misbahan.com)
- Built during Headstarter SWE Residency Sprint 1 