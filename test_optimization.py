import pandas as pd
import numpy as np
from select_best_action import run_optimization_pipeline

# Mock dataset with two dates:
# Date 1: AAPL and MSFT have days_to_earnings=1, rest do not
# Date 2: No earnings-day-1 stocks (full budget should go to knapsack)
data = {
    "Ticker":               ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA",   "NVDA", "META", "NFLX"],
    "Stock_Snapshot_Date":  ["2024-01-10"] * 5 +                        ["2024-01-11"] * 3,
    "chosen_action":        ["call", "call", "put",  "call", "put",     "call", "put",  "call"],
    "selection_score":      [0.75,   0.80,   0.65,   0.70,   0.60,     0.85,   0.72,   0.68],
    "prob_Beat":            [0.75,   0.80,   0.65,   0.70,   0.60,     0.85,   0.72,   0.68],
    "total_investment":     [10000,  15000,  8000,   12000,  9000,     20000,  11000,  7000],
    "Call_Option_Price":    [8.5,    12.0,   0.0,    9.5,    0.0,      14.0,   0.0,    6.5],
    "Put_Option_Price":     [0.0,    0.0,    6.0,    0.0,    7.5,      0.0,    9.0,    0.0],
    "days_to_earnings":     [1,      1,      3,      5,      10,       4,      2,      7],
    "Stock_Price":          [180,    370,    140,    175,    250,      500,    300,    450],
    "Call_Option_Strike":   [185,    375,    0,      180,    0,        510,    0,      460],
    "evaluation_status":    ["EVALUATED"] * 8,
}
df = pd.DataFrame(data)

print("=" * 60)
print("INPUT DATA")
print("=" * 60)
print(df[["Ticker", "Stock_Snapshot_Date", "chosen_action", "days_to_earnings",
          "total_investment", "Call_Option_Price", "Put_Option_Price", "selection_score"]].to_string(index=False))

result = run_optimization_pipeline(
    source_df=df,
    threshold=0.5,
    budget=95000.0,
    max_investment_fraction=0.4,
    max_portfolios=10,
    evaluated_only=False,
)

print("\n" + "=" * 60)
print("OPTIMIZATION RESULT")
print("=" * 60)
print(result[["lookupvalue", "snapshot_date", "option_type", "days_to_earnings",
              "Investment", "options_price", "prediction_prob"]].to_string(index=False))

print("\n" + "=" * 60)
print("BUDGET SPLIT VERIFICATION")
print("=" * 60)
for date, grp in result.groupby("snapshot_date"):
    earnings = grp[pd.to_numeric(grp["days_to_earnings"], errors="coerce") == 1]
    rest = grp[pd.to_numeric(grp["days_to_earnings"], errors="coerce") != 1]
    print(f"\nDate: {date}")
    print(f"  Earnings-day-1 stocks : {list(earnings['lookupvalue'])}  | total invested: ${earnings['Investment'].sum():,.0f}")
    print(f"  Rest stocks           : {list(rest['lookupvalue'])}  | total invested: ${rest['Investment'].sum():,.0f}")
    print(f"  Half budget (50%)     : $47,500")
    print(f"  Earnings <= half?     : {earnings['Investment'].sum() <= 47500}")
    print(f"  Rest <= half?         : {rest['Investment'].sum() <= 47500}")
