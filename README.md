# Freight Rate Prediction

Predicts the dollar rate of 12,000 truckload shipments (Nov to Dec 2025) and a fixed-lane December 2025 rate chart. The model is gradient boosting (scikit-learn HistGradientBoostingRegressor) trained on Jan to Oct 2025 loads.

## Results (time-based check: train Jan-Aug, test Sep-Oct)

| Method | MAE | MAPE | RMSE |
|---|---|---|---|
| Baseline (quote_signal x distance) | $246.02 | 12.11% | $711.80 |
| Gradient boosting model | $132.83 | 5.48% | $640.68 |

The full write-up is in report/report.pdf.

## Setup and run

Requires Python 3.10 or newer. Windows PowerShell commands are shown first, and macOS/Linux equivalents follow.

powershell
git clone https://github.com/Collegekid-coder/freight_rate_prediction.git
cd freight-rate-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv


macOS/Linux: activate with source .venv/bin/activate and use / in paths (python src/train.py).

Expected result: train.py finishes in under a minute. score.py prints
Validated 12,000 final predictions. and Validated 31 fixed December predictions., and writes scorer_results/candidate_december.png.

## What the code does (src/train.py)

1. Loads the raw CSVs (they are never edited).
2. Cleans inputs: negative weights are made positive, blank weights are filled with the median for the trailer type, and blank market_index values are filled with the same-day median.
3. Builds features: distance, straight-line distance, weight, market signals, day of week, coordinates, trailer type, and "was blank" flags.
4. Validates with a time-based split: train on Jan-Aug, test on Sep-Oct, and compare with the baseline.
5. Refits on all Jan-Oct data and predicts the 12,000 validation loads, saved as validation_predictions.csv.
6. Predicts the 31 December chart rows, using each day's median market_index and quote_signal from data/validation.csv, and fills predicted_rate in data/december_chart_inputs.csv.

The random seed is fixed (42), so results are reproducible.

## Files

| Path | Purpose |
|---|---|
| src/train.py | Full pipeline: cleaning, validation, model, both prediction files |
| src/inspect_data.py | Data inventory (shapes, dtypes, missing values, date ranges) |
| src/data_quality.py | Data-quality audit |
| src/check_december.py | Checks the daily inputs behind the Dec 31 uptick |
| notes/ | Data inventory, quality report and training log outputs |
| data/ | Provided input files (december_chart_inputs.csv is filled in by train.py) |
| validation_predictions.csv | Final predictions (load_id,predicted_rate) |
| scorer_results/candidate_december.png | December chart from score.py |
| report/report.docx | Approach, validation and chart report |
| score.py | Provided scorer (validates files, draws the chart) |