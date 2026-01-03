# 🏎️ Formula 1 2025 — Performance Analysis & Points Prediction

## 📌 Overview

This project is a complete statistical and machine learning analysis of the 2025 Formula 1 season, focused on understanding what actually drives race performance and building a clean, defensible model to predict points scored per race.

Instead of chasing overly complex models, the project emphasizes:
- domain understanding of Formula 1
- careful feature engineering
- strict avoidance of data leakage
- step-by-step validation of modeling decisions

The goal is not just prediction — but understanding why predictions make sense.

---

## 🎯 Objectives

This project aims to answer a few core questions:

- Does recent driver form matter more than season averages?
- Does consistency influence race outcomes?
- How important is qualifying performance relative to race execution?
- Does team momentum affect individual drivers?
- Can machine learning improve accuracy beyond simple baselines?
- Where should model complexity realistically stop?

---

## 📊 Data Source

All data is collected using the FastF1 Python library.

Season Used:
- 2025 Formula 1 Season (completed)

Data Includes:
- Race results (positions, grid, points)
- Qualifying results (position, delta to pole, Q3 appearance)
- Driver and team identifiers

All data is cached locally to ensure reproducibility.

---

## 🗂️ Project Structure

```
F1_Project/
├── data/
│   ├── raw/
│   │   ├── race_results_2025.csv
│   │   ├── qualifying_results_2025.csv
│   │   └── cache/
│   └── processed/
│       └── form_features_2025.csv
│
├── notebooks/
│   └── 01-form_eda_2025.ipynb
│
├── src/
│   ├── features/
│   │   └── driver_team_form.py
│   │
│   ├── models/
│   │   ├── baseline_points.py
│   │   ├── tree_baseline_points.py
│   │   ├── gbm_points.py
│   │   ├── interpret_baseline.py
│   │   └── gbm_feature_importance.py
│   │
│   ├── fetch_race_data.py
│   └── fetch_qualifying.py
│
├── .gitignore
├── requirements.txt
└── README.md
```


---

## 📁 Detailed Folder & File Guide

This section explains the purpose of each folder and key file so the project is easy to navigate, even for beginners.

### `data/`
Contains all datasets used in the project.

- **`data/raw/`**
  - `race_results_2025.csv`  
    Raw race-level results for the 2025 season (positions, grid, points).
  - `qualifying_results_2025.csv`  
    Qualifying session data including grid position and delta to pole.
  - `cache/`  
    Local FastF1 cache used to speed up data fetching (ignored by Git).

- **`data/processed/`**
  - `form_features_2025.csv`  
    Final engineered dataset containing driver form, team momentum, and qualifying features used for modeling.

---

### `notebooks/`
Contains analysis notebooks.

- **`01-form_eda_2025.ipynb`**  
  Exploratory Data Analysis notebook.  
  Used to validate assumptions, visualize trends, and confirm that engineered features capture meaningful racing behavior.

---

### `src/`
Houses all production-style Python scripts.

#### `src/features/`
- **`driver_team_form.py`**  
  Builds rolling driver and team form features (last 5 races), ensuring no data leakage by using only past information.

---

#### `src/models/`
Contains all modeling scripts.

- **`baseline_points.py`**  
  Linear regression baseline using form and qualifying features.
- **`tree_baseline_points.py`**  
  Decision tree model to capture non-linear relationships.
- **`gbm_points.py`**  
  Final Gradient Boosting model used for evaluation and comparison.
- **`interpret_baseline.py`**  
  Interprets linear model coefficients for explainability.
- **`gbm_feature_importance.py`**  
  Extracts and displays feature importance from the GBM model.

---

#### `src/fetch_race_data.py`
Fetches race results for the 2025 season using FastF1 and saves them to `data/raw/`.

#### `src/fetch_qualifying.py`
Fetches qualifying session data and qualifying pace metrics for the 2025 season.

---

### Root-level files

- **`.gitignore`**  
  Specifies files and folders that should not be tracked (virtual environment, cache, checkpoints).
- **`requirements.txt`**  
  Lists all Python dependencies required to run the project.
- **`README.md`**  
  Project documentation explaining the motivation, structure, and usage of the repository.

---

## ⚙️ Setup Instructions

1. Clone the repository
git clone https://github.com/Dev-AniketNayak99/F1_Project.git
cd F1_Project

2. Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

---

## 🔄 Running the Project

Fetch race data:
python src/fetch_race_data.py

Fetch qualifying data:
python src/fetch_qualifying.py

Build form features:
python src/features/driver_team_form.py

---

## 🔍 Exploratory Data Analysis (EDA)

Notebook:
notebooks/01-form_eda_2025.ipynb

Key findings:
- Recent driver form strongly correlates with race finish
- Consistency reduces volatility
- Positions gained reflect race execution quality
- Team momentum affects driver performance
- Delta to pole is more informative than grid rank

---

## 🤖 Modeling Strategy

Target:
Points scored per race

Validation:
TimeSeriesSplit to preserve race chronology and avoid leakage.

Models Evaluated:
- Linear regression
- Decision tree
- Gradient boosting (final model)

---

## 🏆 Final Model

Gradient Boosting achieved the best performance with balanced accuracy and interpretability.

Key drivers:
- Qualifying position
- Recent driver points
- Delta to pole
- Team momentum

---

## ⚠️ Limitations

- DNFs introduce randomness
- Weather not modeled
- Single-season scope

---

## 🔮 Future Improvements

- Multi-season modeling
- Track-specific features
- Weather and safety car data
- Uncertainty estimation

---

## 🧠 Key Takeaway

Good machine learning is about clarity, strong features, and knowing when to stop.

This project is designed to be educational, reproducible, and interview-ready.
