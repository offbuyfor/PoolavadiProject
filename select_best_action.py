import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

try:
    import pulp
except Exception:
    pulp = None


KEY_COLS = ["Ticker", "Stock_Snapshot_Date"]
# Target output schema for FOR_EXTERNALfinal_portfolio_optimization.
# Any missing fields are filled with NULL so BigQuery writes remain stable.
FINAL_OPT_SCHEMA_COLS = [
    "option_type",
    "lookupvalue",
    "snapshot_date",
    "earnings_date",
    "calls_exDate",
    "Average_True_Range",
    "Average_Volume",
    "Relative_Volume",
    "Volume",
    "calls_OpenInterest",
    "Volatility_Week",
    "Beta",
    "Close_Price",
    "options_price",
    "price_gap",
    "calls_strike",
    "days_to_earnings",
    "WinorLoss",
    "prediction_prob",
    "expected_return_based_on_prob",
    "Investment",
    "expected_profit",
    "actual_profit_loss",
    "Threshold",
    "Budget",
    "Max_Investment_Fraction",
    "Max_Portfolios",
]


def parse_args() -> argparse.Namespace:
    # CLI supports two stages:
    # 1) choose one best action per ticker/date
    # 2) optional optimization + external schema output
    parser = argparse.ArgumentParser(description="Select one best action (call/put) per ticker and snapshot date")

    # Input sources
    parser.add_argument("--input-call-csv", default="", help="CSV output from call model run")
    parser.add_argument("--input-put-csv", default="", help="CSV output from put model run")
    parser.add_argument("--input-bq-table", default="", help="BigQuery table containing both call and put predictions")

    # Selection logic
    parser.add_argument(
        "--score-column",
        default="prob_Beat",
        choices=["prob_Beat", "max_prob"],
        help="Scoring rule: prob_Beat or max_prob (max(prob_Beat, prob_NoBeat))",
    )

    # Outputs
    parser.add_argument("--output-csv", default="", help="Output CSV path for selected best actions")
    parser.add_argument("--output-bq-table", default="", help="Output BigQuery table in project.dataset.table format")
    parser.add_argument(
        "--write-disposition",
        default="WRITE_TRUNCATE",
        choices=["WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"],
        help="BigQuery write mode when --output-bq-table is used",
    )
    parser.add_argument("--run-optimization", action="store_true", help="Run portfolio optimization after best-action selection")
    parser.add_argument(
        "--nextday-bq-table",
        default="modelraja.UPCOMING_EARNINGS_DATASET.finviz_all_stocks_daily_four_pm_snapshot",
        help="BigQuery table with Ticker/Low/High/snapshot_date_time for win-loss evaluation",
    )
    parser.add_argument(
        "--output-optimization-bq-table",
        default="",
        help="BigQuery table for final optimized portfolio (FOR_EXTERNAL schema)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--budget", type=float, default=95000.0)
    parser.add_argument("--max-investment-fraction", type=float, default=0.4)
    parser.add_argument("--max-portfolios", type=int, default=10)
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame) -> None:
    # Minimal fields required to compare call vs put candidates.
    required = KEY_COLS + ["prediction_flag_type", "prob_Beat", "prob_NoBeat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")


def read_from_csv(call_csv: str, put_csv: str) -> pd.DataFrame:
    # CSV mode expects one file per action model and then concatenates.
    if not call_csv or not put_csv:
        raise ValueError("Both --input-call-csv and --input-put-csv are required for CSV mode")
    call_df = pd.read_csv(call_csv)
    put_df = pd.read_csv(put_csv)
    return pd.concat([call_df, put_df], ignore_index=True)


def read_from_bq(table: str) -> pd.DataFrame:
    # BQ mode expects one table containing both call and put rows.
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery not installed; cannot read BigQuery input")
    client = bigquery.Client()
    query = f"SELECT * FROM `{table}`"
    return client.query(query).to_dataframe()


def select_best_action(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    working = df.copy()

    # Normalize date and probabilities for deterministic ranking.
    working["Stock_Snapshot_Date"] = pd.to_datetime(working["Stock_Snapshot_Date"], errors="coerce")
    working["prob_Beat"] = pd.to_numeric(working["prob_Beat"], errors="coerce")
    working["prob_NoBeat"] = pd.to_numeric(working["prob_NoBeat"], errors="coerce")

    if score_column == "prob_Beat":
        working["selection_score"] = working["prob_Beat"]
    else:
        working["selection_score"] = working[["prob_Beat", "prob_NoBeat"]].max(axis=1)

    # One winner per (Ticker, Stock_Snapshot_Date), highest score first.
    working = working.sort_values(
        by=["Ticker", "Stock_Snapshot_Date", "selection_score", "prob_Beat", "prediction_flag_type"],
        ascending=[True, True, False, False, True],
    )

    selected = working.drop_duplicates(subset=KEY_COLS, keep="first").copy()
    selected["chosen_action"] = selected["prediction_flag_type"]

    # Keep date as date type for BQ DATE compatibility.
    selected["Stock_Snapshot_Date"] = selected["Stock_Snapshot_Date"].dt.date
    return selected


def add_winloss_from_nextday(
    client: "bigquery.Client",
    best_action_table: str,
    nextday_table: str,
) -> pd.DataFrame:
    # Use direct SQL from persisted best-action table (no temp table).
    query = f"""
    SELECT
      p.*,
      CASE
        WHEN p.chosen_action = 'call' THEN IF((p.Stock_Price - next_day.Low) > p.Call_Option_Price, 'Beat', 'NoBeat')
        WHEN p.chosen_action = 'put' THEN IF((next_day.High - p.Stock_Price) > p.Put_Option_Price, 'Beat', 'NoBeat')
        ELSE 'Unknown'
      END AS actual_outcome,
      CASE
        WHEN p.prediction = 'Beat' THEN
          IF(
            CASE
              WHEN p.chosen_action = 'call' THEN IF((p.Stock_Price - next_day.Low) > p.Call_Option_Price, 'Beat', 'NoBeat')
              WHEN p.chosen_action = 'put' THEN IF((next_day.High - p.Stock_Price) > p.Put_Option_Price, 'Beat', 'NoBeat')
              ELSE 'Unknown'
            END = 'Beat',
            1, -1
          )
        ELSE NULL
      END AS win_loss_flag
    FROM `{best_action_table}` AS p
    LEFT JOIN (
      SELECT
        Ticker,
        Low,
        High,
        CASE
          WHEN LENGTH(snapshot_date_time) < 15 THEN PARSE_DATE('%Y-%m-%d', snapshot_date_time)
          ELSE EXTRACT(DATE FROM PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%E*S%Ez', snapshot_date_time))
        END AS snapshot_date
      FROM `{nextday_table}`
    ) AS next_day
      ON p.Ticker = next_day.Ticker
      AND next_day.snapshot_date = DATE_ADD(p.Stock_Snapshot_Date, INTERVAL 1 DAY)
    """
    return client.query(query).to_dataframe()


def solve_knapsack(df_date: pd.DataFrame, budget: float, max_investment_fraction: float, max_portfolios: int) -> pd.DataFrame:
    # Per-date portfolio optimization under budget, concentration, and count constraints.
    if df_date.empty:
        return df_date
    idx = df_date.index.tolist()
    if pulp is None:
        # Fallback greedy selector if PuLP isn't installed.
        ranked = df_date.sort_values("expected_return_based_on_prob", ascending=False)
        chosen = []
        spent = 0.0
        max_each = max_investment_fraction * budget
        for i, row in ranked.iterrows():
            cost = float(row["Investment"])
            if len(chosen) >= max_portfolios:
                break
            if cost <= 0 or cost > max_each:
                continue
            if spent + cost <= budget:
                chosen.append(i)
                spent += cost
        return df_date.loc[chosen].copy()

    problem = pulp.LpProblem("Portfolio_Optimization", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Project", idx, cat=pulp.LpBinary)
    problem += pulp.lpSum(df_date.loc[i, "expected_return_based_on_prob"] * x[i] for i in idx)
    problem += pulp.lpSum(df_date.loc[i, "Investment"] * x[i] for i in idx) <= budget
    for i in idx:
        problem += df_date.loc[i, "Investment"] * x[i] <= max_investment_fraction * budget
    problem += pulp.lpSum(x[i] for i in idx) <= max_portfolios
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return df_date.iloc[0:0].copy()
    chosen = [i for i in idx if x[i].value() == 1]
    return df_date.loc[chosen].copy()


def align_for_external_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Map internal column names to external contract schema and fill absent fields with NULL.
    work = df.copy()
    work["option_type"] = work.get("chosen_action")
    work["lookupvalue"] = work.get("Ticker")
    work["snapshot_date"] = pd.to_datetime(work.get("Stock_Snapshot_Date"), errors="coerce").dt.date
    work["earnings_date"] = work.get("Earnings_Date") if "Earnings_Date" in work.columns else pd.NA
    work["calls_exDate"] = work.get("Call_Option_Expiry_Date") if "Call_Option_Expiry_Date" in work.columns else pd.NA
    work["Close_Price"] = pd.to_numeric(work.get("Stock_Price"), errors="coerce")
    work["options_price"] = np.where(
        work.get("chosen_action").eq("call"),
        pd.to_numeric(work.get("Call_Option_Price"), errors="coerce"),
        pd.to_numeric(work.get("Put_Option_Price"), errors="coerce"),
    )
    work["price_gap"] = pd.to_numeric(work.get("strike_to_close_price_gap"), errors="coerce")
    work["calls_strike"] = pd.to_numeric(work.get("Call_Option_Strike"), errors="coerce")
    work["days_to_earnings"] = pd.to_numeric(work.get("days_to_earnings"), errors="coerce")
    work["WinorLoss"] = pd.to_numeric(work.get("win_loss_flag"), errors="coerce").astype("Int64")
    work["prediction_prob"] = pd.to_numeric(work.get("selection_score"), errors="coerce")
    work["expected_return_based_on_prob"] = pd.to_numeric(work.get("expected_return_based_on_prob"), errors="coerce")
    work["Investment"] = pd.to_numeric(work.get("Investment"), errors="coerce")
    # expected_profit = option premium * 100 (call/put based on chosen action), rounded to 2 decimals.
    work["expected_profit"] = (pd.to_numeric(work["options_price"], errors="coerce") * 100).round(2)
    gap = pd.to_numeric(work.get("price_gap"), errors="coerce")
    loss_component = np.where(
        gap >= 0,
        np.round(gap * -100, 2) - 1,
        np.round(gap * 100, 2) - 1,
    )
    work["actual_profit_loss"] = np.where(
        pd.to_numeric(work["WinorLoss"], errors="coerce") == 1,
        np.round(pd.to_numeric(work["options_price"], errors="coerce") * 100, 2),
        loss_component,
    )

    # Fill required-but-possibly-missing fields as NULL.
    for col in FINAL_OPT_SCHEMA_COLS:
        if col not in work.columns:
            work[col] = pd.NA

    return work[FINAL_OPT_SCHEMA_COLS].copy()


def run_optimization_pipeline(
    client: "bigquery.Client",
    best_action_table: str,
    nextday_table: str,
    threshold: float,
    budget: float,
    max_investment_fraction: float,
    max_portfolios: int,
) -> pd.DataFrame:
    # End-to-end post-selection stage:
    # enrich -> filter by constraints -> optimize per date -> schema align.
    enriched = add_winloss_from_nextday(client, best_action_table, nextday_table)

    enriched["selection_score"] = pd.to_numeric(enriched.get("selection_score"), errors="coerce")
    enriched["Investment"] = pd.to_numeric(enriched.get("total_investment"), errors="coerce")
    enriched["options_price"] = np.where(
        enriched.get("chosen_action").eq("call"),
        pd.to_numeric(enriched.get("Call_Option_Price"), errors="coerce"),
        pd.to_numeric(enriched.get("Put_Option_Price"), errors="coerce"),
    )
    enriched = enriched[enriched["selection_score"] >= threshold].copy()
    enriched = enriched[enriched["Investment"] > 0].copy()
    enriched = enriched[enriched["options_price"] > 0.5].copy()

    enriched["expected_return_based_on_prob"] = enriched["selection_score"] * enriched["Investment"]
    enriched["Threshold"] = float(threshold)
    enriched["Budget"] = int(budget)
    enriched["Max_Investment_Fraction"] = float(max_investment_fraction)
    enriched["Max_Portfolios"] = int(max_portfolios)

    selected_parts = []
    for date, group in enriched.groupby("Stock_Snapshot_Date", dropna=False):
        picked = solve_knapsack(group, budget, max_investment_fraction, max_portfolios)
        if not picked.empty:
            selected_parts.append(picked)

    if selected_parts:
        final = pd.concat(selected_parts, ignore_index=True)
    else:
        final = enriched.iloc[0:0].copy()
    return align_for_external_schema(final)


def write_output(df: pd.DataFrame, output_csv: str, output_bq_table: str, write_disposition: str) -> None:
    # Writes to one or both targets depending on args.
    wrote_any = False

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Wrote selected actions CSV: {output_csv} | rows={len(df)}")
        wrote_any = True

    if output_bq_table:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed; cannot write BigQuery output")
        client = bigquery.Client()
        job = client.load_table_from_dataframe(
            df,
            output_bq_table,
            job_config=bigquery.LoadJobConfig(write_disposition=write_disposition),
        )
        job.result()
        print(f"Wrote selected actions to BigQuery: {output_bq_table} | rows={len(df)}")
        wrote_any = True

    if not wrote_any:
        raise ValueError("Provide at least one output target: --output-csv or --output-bq-table")


def main() -> None:
    args = parse_args()

    using_csv = bool(args.input_call_csv or args.input_put_csv)
    using_bq = bool(args.input_bq_table)
    if using_csv == using_bq:
        raise ValueError("Use exactly one input mode: CSV pair (--input-call-csv + --input-put-csv) OR --input-bq-table")

    # Stage 1: read candidate predictions.
    if using_csv:
        all_preds = read_from_csv(args.input_call_csv, args.input_put_csv)
    else:
        all_preds = read_from_bq(args.input_bq_table)

    # Stage 2: choose one best action per ticker/date.
    ensure_required_columns(all_preds)
    selected = select_best_action(all_preds, args.score_column)
    write_output(selected, args.output_csv, args.output_bq_table, args.write_disposition)

    # Stage 3 (optional): run constrained optimization and write FOR_EXTERNAL table.
    if args.run_optimization:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed; cannot run optimization pipeline")
        if not args.output_optimization_bq_table:
            raise ValueError("--output-optimization-bq-table is required when --run-optimization is set")
        if not args.output_bq_table:
            raise ValueError("--output-bq-table is required with --run-optimization so SQL can read persisted best-action rows")
        client = bigquery.Client()
        optimized = run_optimization_pipeline(
            client=client,
            best_action_table=args.output_bq_table,
            nextday_table=args.nextday_bq_table,
            threshold=args.threshold,
            budget=args.budget,
            max_investment_fraction=args.max_investment_fraction,
            max_portfolios=args.max_portfolios,
        )
        write_output(optimized, "", args.output_optimization_bq_table, args.write_disposition)


if __name__ == "__main__":
    main()
