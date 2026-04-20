"""
Liquidity threshold analysis using training data.
Computes suggested defaults for Volume, Relative_Volume,
Calls_OpenInterest and Market_cap by comparing Beat vs NoBeat
distributions in the training split.
"""

from google.cloud import bigquery
import pandas as pd

PROJECT = "modelraja"
TABLE = "modelraja.MODELS_DATASET.VW_top_gainers_losers_training_data"
LIQUIDITY_COLS = ["Volume", "Relative_Volume", "Calls_OpenInterest", "Market_cap"]

CALL_TARGET = "price_value_delta_above_call_flag"
PUT_TARGET = "price_value_delta_above_put_flag"


def load_data(client: bigquery.Client) -> pd.DataFrame:
    query = f"SELECT * FROM `{TABLE}`"
    print(f"Loading data from {TABLE} ...")
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    print(f"Loaded {len(df)} rows")
    return df


def prep_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Same prep logic as topMoversDatPrep notebooks
    working = df[df[target_col].notna()].copy()
    working["Stock_Snapshot_Date"] = pd.to_datetime(working["Stock_Snapshot_Date"], errors="coerce")
    working = working[working["Stock_Snapshot_Date"].notna()].copy()
    working["snapshot_year"] = working["Stock_Snapshot_Date"].dt.year

    # Assign splits same as notebook (years < 2026 = train/val, 2026 = test)
    working["split"] = "test"
    non_test = working["snapshot_year"] < 2026
    key_ticker = working.loc[non_test, "Ticker"].fillna("").astype(str)
    key_date = working.loc[non_test, "Stock_Snapshot_Date"].dt.strftime("%Y-%m-%d")
    split_key = key_ticker + "_" + key_date
    bucket = (pd.util.hash_pandas_object(split_key, index=False).astype("uint64") % 100).astype(int)
    working.loc[non_test, "split"] = ["train" if b < 85 else "validation" for b in bucket]

    # Coerce liquidity cols to numeric
    for col in LIQUIDITY_COLS:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    return working



def get_beat_p25(raw: pd.DataFrame, target_col: str) -> dict:
    df = prep_data(raw, target_col)
    train = df[df["split"] == "train"]
    beat = train[train[target_col] == "Beat"]
    return {col: beat[col].quantile(0.10) for col in LIQUIDITY_COLS if col in beat.columns}


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    raw = load_data(client)

    call_p25 = {}
    put_p25 = {}

    if CALL_TARGET in raw.columns:
        call_p25 = get_beat_p25(raw, CALL_TARGET)
        print(f"\nCall Beat p25: {call_p25}")
    else:
        print(f"WARNING: {CALL_TARGET} not found in table")

    if PUT_TARGET in raw.columns:
        put_p25 = get_beat_p25(raw, PUT_TARGET)
        print(f"Put  Beat p25: {put_p25}")
    else:
        print(f"WARNING: {PUT_TARGET} not found in table")

    print(f"\n{'='*60}")
    print(f"  SUGGESTED THRESHOLDS (average of call p10 and put p10)")
    print(f"{'='*60}")
    print(f"\n{'Column':<25} {'Call p10':>12} {'Put p10':>12} {'Average':>12}")
    print("-" * 65)
    for col in LIQUIDITY_COLS:
        c = call_p25.get(col)
        p = put_p25.get(col)
        if c is None and p is None:
            print(f"{col:<25}  NOT FOUND")
            continue
        vals = [v for v in [c, p] if v is not None]
        avg = sum(vals) / len(vals)
        c_str = f"{c:.2f}" if c is not None else "N/A"
        p_str = f"{p:.2f}" if p is not None else "N/A"
        print(f"{col:<25} {c_str:>12} {p_str:>12} {avg:>12.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
