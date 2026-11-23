import pandas as pd
from pathlib import Path

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

DATA = Path("data")
names = ["train_test", "validation", "validation_predictions_template", "december_chart_inputs"]
frames = {}

for name in names:
    df = pd.read_csv(DATA / f"{name}.csv")
    frames[name] = df
    print("=" * 80)
    print(name, "shape:", df.shape)
    print("\nDTYPES\n", df.dtypes)
    print("\nHEAD\n", df.head(10).to_string())
    miss = df.isna().sum()
    print("\nMISSING\n", miss[miss > 0] if (miss > 0).any() else "none")
    for c in df.columns:
        if "date" in c.lower():
            d = pd.to_datetime(df[c], errors="coerce")
            print(f"\n{c}: min={d.min()} max={d.max()} unparseable={d.isna().sum()}")

print("=" * 80)
print("COLUMN COMPARISON")
tr, va, dec = (set(frames[k].columns) for k in ["train_test", "validation", "december_chart_inputs"])
print("in train but not validation:", sorted(tr - va))
print("in validation but not train:", sorted(va - tr))
print("in train but not december:", sorted(tr - dec))

ids_match = set(frames["validation"]["load_id"]) == set(frames["validation_predictions_template"]["load_id"])
print("\nvalidation IDs match template:", ids_match)

print("=" * 80)
print("TRAIN DESCRIBE\n", frames["train_test"].describe(include="all").T.to_string())