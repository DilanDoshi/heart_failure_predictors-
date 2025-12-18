# Overview  
Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for an estimated 17.9 million deaths each year—approximately 31% of all deaths worldwide. Heart failure, a common consequence of CVDs, poses a major health burden that requires timely prediction and intervention. This project utilizes a dataset containing 12 clinical features to develop a machine learning model for predicting mortality due to heart failure. Early identification of high-risk patients through such predictive modeling can support proactive disease management and improve patient outcomes.

### See Research_Report.pdf for a full report on the project, desciption of methods, and analysis of results.

**Goal**

Predict whether a patient with heart failure will die during the follow-up period (`DEATH_EVENT` = 1 vs 0) based on 12 clinical variables.

**Type of problem**

- Supervised learning
- Binary classification (DEATH vs SURVIVAL)
- Small tabular dataset (299 rows, 13 columns including target)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd m148_heart_failure_predictors-
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
     ```

5. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost; print('All packages installed successfully!')"
   ```

## Usage

### Running the Notebooks

The project is organized into sequential notebooks:

1. **`01-EDA.ipynb`** - Exploratory Data Analysis
   - Data loading and initial exploration
   - Feature distribution analysis
   - Outlier detection and removal

2. **`02-data-cleaning-preprocessing.ipynb`** - Data Preparation
   - Missing value handling
   - Feature scaling and standardization
   - Train/validation split

3. **`03-ml-and-predictions.ipynb`** - Model Training & Evaluation
   - Baseline model establishment
   - Hyperparameter tuning for multiple models
   - Model comparison and selection

4. **`05-reporting-evaluation.ipynb`** - Final Evaluation & Interpretation
   - Feature importance analysis
   - Final model evaluation
   - Overfitting analysis

### Starting Jupyter Notebook

After activating your virtual environment:

```bash
jupyter notebook
```

Or if using JupyterLab:

```bash
jupyter lab
```

Navigate to the `notebooks/` directory and open the notebooks in order.

### How to Use the Code (step-by-step)

1) Ensure the raw dataset exists at `data/heart_failure_clinical_records_dataset.csv`.
2) Launch Jupyter (`jupyter notebook` or `jupyter lab`) and open `notebooks/01-EDA.ipynb` to inspect the raw data and outlier handling.
3) Run `notebooks/02-data-cleaning-preprocessing.ipynb` to apply preprocessing; it can write a cleaned file to `data/heart_failure_clinical_records_dataset_cleaned.csv` (uncomment the save cell if needed).
4) Run `notebooks/03-ml-and-predictions.ipynb` to train/evaluate classical ML models; metrics/plots are produced in the notebook cells.
5) (Optional) Run `notebooks/04-nn-and-predictions.ipynb` for simple neural-network experiments, if present in your checkout.
6) Run `notebooks/05-reporting-evaluation.ipynb` to aggregate results, review feature importances, and capture final evaluation.

Tips:
- Execute each notebook top-to-bottom to keep intermediate variables consistent.
- If you add new models or preprocessing steps, store any intermediate data in `data/` to keep the repo tidy.

### Project Structure

```
m148_heart_failure_predictors-/
├── data/                          # Dataset files
│   ├── heart_failure_clinical_records_dataset.csv
│   └── heart_failure_clinical_records_dataset_cleaned.csv
├── notebooks/                     # Jupyter notebooks
│   ├── 01-EDA.ipynb
│   ├── 02-data-cleaning-preprocessing.ipynb
│   ├── 03-ml-and-predictions.ipynb
│   └── 05-reporting-evaluation.ipynb
├── helper_funcs/                  # Helper functions
│   └── evals.py                   # Evaluation utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Notes

- The notebooks should be run in sequential order (01 → 02 → 03 → 05)
- Intermediate data files (train/validation splits) are saved in the `data/` directory
- The project uses a validation set as the final test set (no separate test set created) as our data set is small
