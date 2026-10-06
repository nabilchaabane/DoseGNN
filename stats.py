import pandas as pd
import numpy as np

df = pd.read_csv("matrix/dist_error_10f125040_arms_highres.csv")

print(f"rows: {len(df):,}")
print("\ncolumns:", list(df.columns))

# --- censored flag ---
print("\n--- censored ---")
print(df["censored"].value_counts(dropna=False))
print(f"  fraction censored: {df['censored'].mean():.4f}")

# --- cross-tab: censored vs weight==0 ---
print("\n--- censored x (weight == 0) ---")
print(pd.crosstab(df["censored"], df["weight"] == 0))

# --- dose distribution by censored flag ---
print("\n--- true_log_dose by censored ---")
print(df.groupby("censored")["true_log_dose"]
        .describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

# --- error by censored flag (this is the key one) ---
print("\n--- abs_error_decades by censored ---")
print(df.groupby("censored")["abs_error_decades"]
        .describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

# --- distance by censored flag ---
print("\n--- distance_cm by censored ---")
print(df.groupby("censored")["distance_cm"]
        .describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

# --- weight by censored flag ---
print("\n--- weight by censored ---")
print(df.groupby("censored")["weight"]
        .describe(percentiles=[.05, .5, .95]).to_string())