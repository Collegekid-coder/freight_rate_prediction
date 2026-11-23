import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

DATA = Path("data")
train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])

train["rate_per_mile"] = train["posted_rate"] / train["distance"]


def month(df):
    return df["date"].dt.to_period("M")


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# 1. Duplicates and impossible rows
section("1. DUPLICATES AND IMPOSSIBLE ROWS")
check_cols = [c for c in train.columns if c not in ("load_id", "rate_per_mile")]
print("duplicate load_id (train):", train["load_id"].duplicated().sum())
print("duplicate rows ignoring load_id (train):", train.duplicated(subset=check_cols).sum())
print("pickup == delivery (train / validation):",
      (train["pickup"] == train["delivery"]).sum(), "/", (val["pickup"] == val["delivery"]).sum())
print("posted_rate <= 0 (train):", (train["posted_rate"] <= 0).sum())
print("distance <= 0 (train / validation):",
      (train["distance"] <= 0).sum(), "/", (val["distance"] <= 0).sum())

# 2. Weight
section("2. WEIGHT")
for name, df in [("train", train), ("validation", val)]:
    w = df["weight"]
    print(f"{name}: rows={len(w)} missing={w.isna().sum()} negative={(w < 0).sum()} "
          f"zero={(w == 0).sum()} tiny(0<w<1000)={((w > 0) & (w < 1000)).sum()} "
          f"min={w.min()} max={w.max()}")
print("\nabs(negative weights) summary:")
print(train.loc[train["weight"] < 0, "weight"].abs().describe().round(1).to_string())
print("\npositive weights summary:")
print(train.loc[train["weight"] > 0, "weight"].describe().round(1).to_string())

# 3. Missing values: random or patterned?
section("3. MISSING VALUES: RANDOM OR PATTERNED?")
for col in ["weight", "market_index"]:
    print(f"\n{col}: share missing by month (train)")
    print(train[col].isna().groupby(month(train)).mean().round(4).to_string())
    print(f"{col}: share missing by equipment (train)")
    print(train[col].isna().groupby(train["equipment"]).mean().round(4).to_string())
    print(f"{col}: share missing, train vs validation:",
          round(train[col].isna().mean(), 4), "vs", round(val[col].isna().mean(), 4))

# 4. Is market_index a daily level or per-load?
section("4. MARKET_INDEX: DAILY LEVEL OR PER-LOAD?")
within = train.groupby("date")["market_index"].std().mean()
between = train.groupby("date")["market_index"].mean().std()
print(f"average spread of market_index within one day: {within:.4f}")
print(f"spread of the daily averages across days:     {between:.4f}")
print("(if 'within' is much smaller than 'between', a blank can be filled from the same day)")

# 5. Target
section("5. TARGET: posted_rate AND rate_per_mile")
q = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
print(train["posted_rate"].describe(percentiles=q).round(2).to_string())
print()
print(train["rate_per_mile"].describe(percentiles=q).round(3).to_string())
cols = ["load_id", "pickup", "delivery", "distance", "equipment", "weight", "date",
        "quote_signal", "posted_rate", "rate_per_mile"]
print("\n10 highest rate_per_mile:")
print(train.nlargest(10, "rate_per_mile")[cols].to_string())
print("\n10 lowest rate_per_mile:")
print(train.nsmallest(10, "rate_per_mile")[cols].to_string())
print("\nrate_per_mile by equipment:")
print(train.groupby("equipment")["rate_per_mile"].agg(["count", "mean", "median"]).round(3).to_string())

# 6. quote_signal as a predictor
section("6. QUOTE_SIGNAL AS A PREDICTOR (the naive baseline)")
naive = train["quote_signal"] * train["distance"]
err = naive - train["posted_rate"]
print("corr(quote_signal * distance, posted_rate):", round(naive.corr(train["posted_rate"]), 4))
print("ratio actual rate_per_mile / quote_signal:")
print((train["rate_per_mile"] / train["quote_signal"])
      .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(4).to_string())
print(f"naive baseline MAE:  {err.abs().mean():.2f}")
print(f"naive baseline MAPE: {(err.abs() / train['posted_rate']).mean() * 100:.2f}%")
print(f"naive baseline RMSE: {np.sqrt((err ** 2).mean()):.2f}")


# 7. Distance vs coordinates
def haversine_miles(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


section("7. DISTANCE VS COORDINATES")
for name, df in [("train", train), ("validation", val)]:
    geo = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    factor = (df["distance"] / geo).replace([np.inf, -np.inf], np.nan)
    print(f"\n{name}: distance / straight-line miles")
    print(factor.describe(percentiles=[0.01, 0.5, 0.99]).round(3).to_string())

# 8. Cities, coordinates, unseen categories, and the December lane
section("8. CITIES, COORDINATES, DECEMBER LANE")
pu = train[["pickup", "pickup_lat", "pickup_lon"]].rename(
    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})
de = train[["delivery", "delivery_lat", "delivery_lon"]].rename(
    columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
coords = pd.concat([pu, de])
per_city = coords.groupby("city")[["lat", "lon"]].nunique()
print("cities:", len(per_city), "| cities with more than one coordinate pair:",
      int(((per_city["lat"] > 1) | (per_city["lon"] > 1)).sum()))
for col in ["pickup", "delivery", "equipment"]:
    print(f"{col} values in validation but never in train:", sorted(set(val[col]) - set(train[col])))
for city in ["Lexington", "Fort Wayne"]:
    rows = coords[coords["city"] == city]
    print(f"{city}: seen {len(rows)} times in train; coordinates:",
          rows[["lat", "lon"]].drop_duplicates().values.tolist())
lane = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
print("Lexington -> Fort Wayne loads in train:", len(lane))
if len(lane):
    print(lane["distance"].describe().round(1).to_string())

# 9. Drift over time
section("9. DRIFT OVER TIME")
print("TRAIN by month")
print(train.groupby(month(train)).agg(
    loads=("load_id", "count"), market_index=("market_index", "mean"),
    quote_signal=("quote_signal", "mean"), rate_per_mile=("rate_per_mile", "mean"),
    distance=("distance", "mean")).round(3).to_string())
print("\nVALIDATION by month")
print(val.groupby(month(val)).agg(
    loads=("load_id", "count"), market_index=("market_index", "mean"),
    quote_signal=("quote_signal", "mean"), distance=("distance", "mean")).round(3).to_string())
print("\nTRAIN mean rate_per_mile by day of week")
print(train.groupby(train["date"].dt.day_name())["rate_per_mile"].mean().round(3).to_string())
print("\nVALIDATION numeric ranges")
print(val[["distance", "weight", "market_index", "quote_signal"]].describe().round(3).to_string())