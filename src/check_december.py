"""Checks why December 31 spikes: looks at the daily inputs fed to the model."""
import pandas as pd

# Load the 12,000 validation loads, parsing the date column as real dates.
val = pd.read_csv("data/validation.csv", parse_dates=["date"])

# Keep December only.
dec = val[val["date"] >= "2025-12-01"]

# For each day: how many loads there are, and the typical market_index / quote_signal.
daily = dec.groupby("date").agg(
    loads=("load_id", "count"),
    market_index=("market_index", "median"),
    quote_signal=("quote_signal", "median"),
)

# Print the table so we can compare Dec 31 against other days.
print(daily.round(3).to_string())