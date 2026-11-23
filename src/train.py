"""
train.py - freight rate prediction pipeline.

What this script does, in order:
  1. Loads the raw CSVs (never edits them).
  2. Cleans the inputs (fixes negative weights, fills blanks).
  3. Builds features (the columns the model learns from).
  4. Checks the model honestly: train on Jan-Aug, test on Sep-Oct.
  5. Compares the model to the simple baseline (quote_signal x distance).
  6. Refits on all Jan-Oct data and predicts the 12,000 validation loads.
  7. Predicts the 31 fixed December loads for the chart.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------- settings
SEED = 42                                # fixed seed so every run gives the same answer
DATA = Path("data")                      # folder holding the CSV files
SPLIT_DATE = pd.Timestamp("2025-09-01")  # train before this date, test on/after it
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]  # the 3 trailer types seen in training


# ---------------------------------------------------------------- helpers
def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line distance in miles between two latitude/longitude points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)           # degrees -> radians
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))             # 3958.8 = Earth radius in miles


def report_errors(label, actual, predicted):
    """Print three error scores so we can compare methods on the same footing."""
    error = predicted - actual
    mae = np.abs(error).mean()                            # average miss in dollars
    mape = (np.abs(error) / actual).mean() * 100          # average miss as % of the true price
    rmse = np.sqrt((error ** 2).mean())                   # like MAE but punishes big misses more
    print(f"{label:<28} MAE ${mae:8.2f} | MAPE {mape:5.2f}% | RMSE ${rmse:8.2f}")
    return mae, mape


# ---------------------------------------------------------------- cleaning
def clean_inputs(df, weight_medians, market_fallback):
    """Fix bad values and fill blanks. Works on a copy; the raw data stays untouched."""
    df = df.copy()

    # Negative weights are impossible and look like sign mistakes -> make them positive.
    df["weight"] = df["weight"].abs()

    # Remember which weights were blank (the blank itself may carry information),
    # then fill them with the typical weight for that trailer type.
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))

    # market_index is a daily level, so a blank is best filled from other loads on
    # the same day. If a whole day is blank, fall back to the overall typical value.
    df["market_missing"] = df["market_index"].isna().astype(int)
    same_day = df.groupby("date")["market_index"].transform("median")
    df["market_index"] = df["market_index"].fillna(same_day).fillna(market_fallback)
    return df


# ---------------------------------------------------------------- features
def make_features(df):
    """Turn a cleaned table into the numeric columns the model learns from."""
    straight = haversine_miles(df["pickup_lat"], df["pickup_lon"],
                               df["delivery_lat"], df["delivery_lon"])
    X = pd.DataFrame({
        "distance": df["distance"],                       # road miles
        "straight_line_miles": straight,                  # as-the-crow-flies miles
        "distance_ratio": df["distance"] / straight,      # flags rows where distance looks inconsistent
        "weight": df["weight"],
        "weight_missing": df["weight_missing"],
        "market_index": df["market_index"],
        "market_missing": df["market_missing"],
        "quote_signal": df["quote_signal"],
        "day_of_week": df["date"].dt.dayofweek,           # 0=Mon..6=Sun. NOT month: December is unseen
        "pickup_lat": df["pickup_lat"], "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"], "delivery_lon": df["delivery_lon"],
    })
    # One yes/no column per trailer type (models need numbers, not words).
    for eq in EQUIPMENT_TYPES:
        X["equip_" + eq] = (df["equipment"] == eq).astype(int)
    return X


# ---------------------------------------------------------------- model
def fit_model(fit_df):
    """Learn from labelled rows. Returns everything needed to predict later."""
    # Typical values come ONLY from the rows we learn from, so nothing leaks in.
    weight_medians = fit_df["weight"].abs().groupby(fit_df["equipment"]).median().to_dict()
    market_fallback = fit_df["market_index"].median()

    clean = clean_inputs(fit_df, weight_medians, market_fallback)
    X = make_features(clean)

    # Target = log of dollars PER MILE. Per-mile prices are comparable between short
    # and long loads, and the log squeezes the extreme tail so it dominates less.
    y = np.log(fit_df["posted_rate"] / fit_df["distance"])

    model = HistGradientBoostingRegressor(
        loss="absolute_error",    # robust to outliers: big misses don't dominate
        max_iter=300,             # number of boosting rounds (trees)
        learning_rate=0.08,       # how big a correction each tree is allowed to make
        max_leaf_nodes=31,        # size limit per tree
        min_samples_leaf=40,      # each leaf needs 40+ loads, which avoids memorising
        early_stopping=False,     # fixed number of rounds -> reproducible results
        random_state=SEED,
    )
    model.fit(X, y)
    return {"model": model, "weight_medians": weight_medians, "market_fallback": market_fallback}


def predict_rates(bundle, df):
    """Predict the dollar rate for each row of df."""
    clean = clean_inputs(df, bundle["weight_medians"], bundle["market_fallback"])
    X = make_features(clean)
    per_mile = np.exp(bundle["model"].predict(X))         # undo the log -> dollars per mile
    dollars = per_mile * clean["distance"].values         # per-mile price x miles = total price
    return np.clip(dollars, 1.0, None)                    # scorer needs every rate > 0


# ---------------------------------------------------------------- 1. load data
train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
template = pd.read_csv(DATA / "validation_predictions_template.csv")
dec_raw = pd.read_csv(DATA / "december_chart_inputs.csv")   # kept as text to preserve its format

# ---------------------------------------------------------------- 2. cleaning summary (for the report)
print("DATA CLEANING SUMMARY")
print(f"negative weights fixed  - train: {(train['weight'] < 0).sum()}, validation: {(val['weight'] < 0).sum()}")
print(f"blank weights filled    - train: {train['weight'].isna().sum()}, validation: {val['weight'].isna().sum()}")
print(f"blank market_index fill - train: {train['market_index'].isna().sum()}, validation: {val['market_index'].isna().sum()}")

# ---------------------------------------------------------------- 3. honest check: time-based split
fit_part = train[train["date"] < SPLIT_DATE]     # Jan-Aug: the model learns from these
test_part = train[train["date"] >= SPLIT_DATE]   # Sep-Oct: "the future" the model has never seen
print(f"\nTIME-BASED CHECK: train on {len(fit_part)} loads (Jan-Aug), test on {len(test_part)} loads (Sep-Oct)")

check_bundle = fit_model(fit_part)
check_pred = predict_rates(check_bundle, test_part)
actual = test_part["posted_rate"].values

# Baseline: the market's per-mile quote x distance (no learning at all).
baseline = (test_part["quote_signal"] * test_part["distance"]).values
base_mae, base_mape = report_errors("Baseline (quote x distance)", actual, baseline)
model_mae, model_mape = report_errors("Gradient boosting model", actual, check_pred)
print(f"Improvement over baseline: MAE {100 * (base_mae - model_mae) / base_mae:.1f}% | "
      f"MAPE {100 * (base_mape - model_mape) / base_mape:.1f}%")

# ---------------------------------------------------------------- 4. final model on ALL labelled data
print("\nFitting final model on all Jan-Oct loads...")
final_bundle = fit_model(train)

# ---------------------------------------------------------------- 5. predict the 12,000 validation loads
val_pred = predict_rates(final_bundle, val)
predictions = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val_pred.round(2)})
# Follow the template's row order exactly, and make sure no load is missing.
predictions = template[["load_id"]].merge(predictions, on="load_id", how="left")
assert predictions["predicted_rate"].notna().all(), "some loads have no prediction"
assert len(predictions) == 12000, "expected exactly 12,000 rows"
predictions.to_csv("validation_predictions.csv", index=False)
print("Saved validation_predictions.csv:", predictions.shape,
      "| min", predictions["predicted_rate"].min(), "| max", predictions["predicted_rate"].max())

# ---------------------------------------------------------------- 6. December chart predictions
dec = dec_raw.copy()
dec["date"] = pd.to_datetime(dec_raw["date"])

# The chart file has no market_index / quote_signal, but both are INPUT columns (not answers).
# Use each December day's typical value from that day's loads in validation.csv.
daily = val.groupby("date")[["market_index", "quote_signal"]].median()
dec["market_index"] = dec["date"].map(daily["market_index"]).fillna(val["market_index"].median())
dec["quote_signal"] = dec["date"].map(daily["quote_signal"]).fillna(val["quote_signal"].median())

# The chart file has no coordinates, so look each city up from the training data.
cities = pd.concat([
    train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
]).drop_duplicates("city").set_index("city")
dec["pickup_lat"] = dec["pickup"].map(cities["lat"])
dec["pickup_lon"] = dec["pickup"].map(cities["lon"])
dec["delivery_lat"] = dec["delivery"].map(cities["lat"])
dec["delivery_lon"] = dec["delivery"].map(cities["lon"])

dec_raw["predicted_rate"] = predict_rates(final_bundle, dec).round(2)   # fill the blank column
dec_raw.to_csv(DATA / "december_chart_inputs.csv", index=False)         # same 7 columns, same order
print("\nDECEMBER PREDICTIONS (fixed lane: Lexington -> Fort Wayne)")
print(dec_raw[["date", "predicted_rate"]].to_string(index=False))