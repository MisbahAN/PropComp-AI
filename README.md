# Property Recommendation System

A machine learning system that identifies and explains the best comparable properties for real estate appraisals. Built during the first sprint of Headstarter SWE Residency.

## Project Overview

This system processes property appraisal data through several stages:
1. Data cleaning and standardization
2. Feature engineering
3. Training a ranking model
4. Generating human-readable explanations for recommendations

## 📁 Project Structure

### Directory Layout
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

### Generated Files & Their Purpose

- `data/raw/appraisals_dataset.json`
  - Raw input JSON of appraisals with subject, comps, and candidate properties.

- `data/cleaned/cleaned_appraisals_dataset.json`
  - Output of clean_initial_data.py: all fields (age, GLA, lot, rooms, baths) standardized to numeric values.

- `data/engineered/feature_engineered_appraisals_dataset.json`
  - Output of features.py: adds engineered fields (diffs, binary flags) to each record.

- `data/training/training_data.csv`
  - Output of training_data.py: a flat table where each row is one candidate property, labeled is_comp (1 = true comp, 0 = not), with all signed and absolute feature diffs plus flags.

- `models/xgb_rank_model.json`
  - Trained XGBoost pairwise ranking model saved by train_model.py.

- `outputs/top3_gpt_explanations.csv`
  - Explains top 3 comps per appraisal using SHAP + GPT—in case you re-enable that step.

## 🛠️ Implementation Details

### Features Used

The model uses the following features, all computed as differences between subject and candidate properties:

#### Physical Characteristics
- **Gross Living Area (GLA)**
  - `gla_diff`: Difference in square footage
  - `abs_gla_diff`: Absolute difference in square footage

- **Lot Size**
  - `lot_size_sf_diff`: Difference in square feet
  - `abs_lot_size_sf_diff`: Absolute difference in square feet

- **Age**
  - `effective_age_diff`: Difference in effective age
  - `subject_age_diff`: Difference in subject age
  - `abs_effective_age_diff`: Absolute difference in effective age
  - `abs_subject_age_diff`: Absolute difference in subject age

- **Rooms**
  - `room_count_diff`: Difference in total rooms
  - `bedrooms_diff`: Difference in number of bedrooms
  - `abs_room_count_diff`: Absolute difference in total rooms
  - `abs_bedrooms_diff`: Absolute difference in bedrooms

- **Bathrooms**
  - `bath_score_diff`: Difference in bathroom score
  - `full_baths_diff`: Difference in full bathrooms
  - `half_baths_diff`: Difference in half bathrooms
  - `abs_bath_score_diff`: Absolute difference in bathroom score
  - `abs_full_bath_diff`: Absolute difference in full bathrooms
  - `abs_half_bath_diff`: Absolute difference in half bathrooms

#### Property Type & Timing
- `same_property_type`: Binary flag (1/0) if property types match
- `sold_recently`: Binary flag (1/0) if sold within 90 days of subject's effective date

### Data Processing Pipeline

1. `clean_initial_data.py` (375 lines)
   - Standardizes property data
   - Handles missing values
   - Normalizes property features (ages, sizes, rooms)
   - Outputs cleaned data to `data/cleaned/`

2. `features.py` (454 lines)
   - Adds engineered features
   - Implements property type matching
   - Creates time-based flags
   - Outputs enhanced data to `data/engineered/`
   - **Modify this file to change the features used in the model**

3. `training_data.py` (218 lines)
   - Converts data to ML format
   - Creates positive/negative examples
   - Prepares data for model training
   - Outputs to `data/training/`

4. `train_model.py` (121 lines)
   - Trains XGBoost ranking model
   - Implements cross-validation
   - Saves model to `models/`
   - **Modify this file to change model parameters (learning rate, depth, etc.)**

5. `top3_explanations.py` (171 lines)
   - Generates human-readable explanations
   - Uses SHAP for feature importance
   - Integrates with GPT for natural language explanations
   - Outputs to `outputs/`
   - **Modify this file to change the explanation format and GPT prompts**

## 🔑 Key Concepts Cheat-Sheet

### 1. Text Processing

#### a) Regular Expressions
Extract structured data from messy strings.
```python
import re
# Extract year from "built in 2023"
match = re.search(r"(\d{1,4})", "built in 2023")
year  = int(match.group(1))  # → 2023
```

#### b) Tokenization
Split a cleaned string into meaningful parts.
```python
# Separate number + unit
val    = "1200 sqft"
tokens = val.split()        # → ["1200", "sqft"]
```

#### c) Fuzzy String Matching
Map free-text to a small set of known categories.
```python
from fuzzywuzzy import process
CANONICAL_TYPES = ["Townhouse", "Detached", ...]
match, score = process.extractOne("semi detached", CANONICAL_TYPES)
# If score ≥ 80, accept `match`; otherwise None
```

### 2. Data Splitting & Grouping

#### a) Train/Test Split
Hold out data for honest evaluation.
```python
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
```

#### b) Group Creation
Tell the ranker which rows belong to each query (appraisal).
```python
# Count candidates per appraisal (orderID)
groups = df_train.groupby("orderID").size().to_list()
```

### 3. XGBoost Ranking

#### a) DMatrix
XGBoost's optimized input format (features + labels + groups).
```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(groups)
```

#### b) Pairwise Ranking Objective
Learn to order items by comparing pairs within each group.
```python
params = {
    "objective":  "rank:pairwise",
    "eval_metric":"ndcg",        # Normalized Discounted Cumulative Gain
    "eta":         0.1,          # learning rate
    "max_depth":   6
}
model = xgb.train(params, dtrain, num_boost_round=100)
```

### 4. Evaluation Metric: Top-K Precision
"Of my top K suggestions, how many are correct?"
```python
def precision_at_k(group, k):
    X      = xgb.DMatrix(group[feature_cols])
    group["score"] = model.predict(X)
    topk   = group.nlargest(k, "score")
    return topk["label"].sum() / k

# Example: average Precision@3 across all test appraisals
overall = df_test.groupby("orderID").apply(lambda g: precision_at_k(g, 3))
print("Avg Precision@3:", overall.mean())
```

### 5. SHAP Explanations
Break down each model prediction into feature-level contributions.
```python
import shap

# Wrap the trained model
def model_predict(X_df):
    return model.predict(xgb.DMatrix(X_df))

# Use background data for reference
explainer = shap.Explainer(model_predict, df[feature_cols])

# For a single candidate row:
row_df   = candidate_row[feature_cols].to_frame().T
shap_vals = explainer(row_df).values[0]          # one SHAP value per feature
shap_items = list(zip(feature_cols, shap_vals))
# Positive contributors
pos = [(f, v) for f, v in shap_items if v > 0]
# Negative contributors
neg = [(f, v) for f, v in shap_items if v < 0]
```

## 🚀 Getting Started

### Prerequisites

1. Python 3.8 or higher
2. OpenAI API key (for generating explanations)
   - Sign up at [OpenAI Platform](https://platform.openai.com)
   - Create an API key in your account settings
   - Copy `.env.example` to `.env` and add your API key:
     ```bash
     cp .env.example .env
     # Edit .env and add your OpenAI API key
     ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Property-Recommendation-System.git
   cd Property-Recommendation-System
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Setup

1. Create a `.env` file in the project root:
   ```bash
   touch .env  # On macOS/Linux
   # OR
   type nul > .env  # On Windows
   ```

2. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Verify the setup:
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('API Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
   ```

### Running the Pipeline

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

### Requirements

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