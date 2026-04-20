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
    "Earnings_Date",
    "Option_Expiry_Date",
    "Average_True_Range",
    "Average_Volume",
    "Relative_Volume",
    "Volume",
    "calls_OpenInterest",
    "Volatility_Week",
    "Beta",
    "Market_Cap",
    "Close_Price",
    "options_price",
    "price_gap",
    "calls_strike",
    "days_to_earnings",
    "WinorLoss",
    "evaluation_status",
    "prediction_prob",
    "expected_return_based_on_prob",
    "Investment",
    "expected_profit",
    "actual_profit_loss",
    "eod_nextday_High",
    "eod_nextday_Low",
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
        choices=["prob_Beat"],
        help="Scoring rule used for ranking candidates (fixed to prob_Beat).",
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
        help="(Backward-compatible) BigQuery table for optimized portfolio; used as evaluated table if specific outputs are not provided",
    )
    parser.add_argument(
        "--output-optimization-live-bq-table",
        default="",
        help="BigQuery table for LIVE optimized portfolio output (FOR_EXTERNAL schema)",
    )
    parser.add_argument(
        "--output-evaluated-bq-table",
        default="",
        help="BigQuery table for evaluated signals (actual_outcome/win_loss_flag/evaluation_status)",
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


def normalize_market_cap_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Market_Cap" not in out.columns and "Market_cap" in out.columns:
        out["Market_Cap"] = out["Market_cap"]
    if "Market_Cap" in out.columns:
        out["Market_Cap"] = pd.to_numeric(out["Market_Cap"], errors="coerce")
    return out


def select_best_action(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    working = df.copy()

    # Normalize date and probabilities for deterministic ranking.
    working["Stock_Snapshot_Date"] = pd.to_datetime(working["Stock_Snapshot_Date"], errors="coerce")
    working["prob_Beat"] = pd.to_numeric(working["prob_Beat"], errors="coerce")
    working["prob_NoBeat"] = pd.to_numeric(working["prob_NoBeat"], errors="coerce")

    # Selection score is intentionally fixed to prob_Beat.
    working["selection_score"] = working["prob_Beat"]

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
      SAFE_CAST(next_day.High AS FLOAT64) AS joined_eod_nextday_High,
      SAFE_CAST(next_day.Low AS FLOAT64) AS joined_eod_nextday_Low,
      CASE
        WHEN p.chosen_action = 'call' THEN IF(IFNULL((p.Stock_Price - next_day.Low) > p.Call_Option_Price, FALSE), 'Beat', 'NoBeat')
        WHEN p.chosen_action = 'put' THEN IF(IFNULL((next_day.High - p.Stock_Price) > p.Put_Option_Price, FALSE), 'Beat', 'NoBeat')
        ELSE 'Unknown'
      END AS actual_outcome,
      CASE
        WHEN LOWER(TRIM(CAST(p.prediction AS STRING))) IN ('beat', '1', 'true') THEN
          IF(
            CASE
              WHEN p.chosen_action = 'call' THEN IF(IFNULL((p.Stock_Price - next_day.Low) > p.Call_Option_Price, FALSE), 'Beat', 'NoBeat')
              WHEN p.chosen_action = 'put' THEN IF(IFNULL((next_day.High - p.Stock_Price) > p.Put_Option_Price, FALSE), 'Beat', 'NoBeat')
              ELSE 'Unknown'
            END = 'Beat',
            1, -1
          )
        ELSE NULL
      END AS win_loss_flag,
      CASE
        WHEN next_day.Low IS NULL OR next_day.High IS NULL THEN 'PENDING_NEXT_DAY_DATA'
        ELSE 'EVALUATED'
      END AS evaluation_status
    FROM (
      SELECT *
      FROM `{best_action_table}`
      WHERE LOWER(TRIM(CAST(prediction AS STRING))) IN ('beat', '1', 'true')
    ) AS p
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
    def pick(*cols):
        for c in cols:
            if c in work.columns:
                return work[c]
        return pd.Series(pd.NA, index=work.index)

    work["option_type"] = work.get("chosen_action")
    work["lookupvalue"] = work.get("Ticker")
    work["snapshot_date"] = pd.to_datetime(work.get("Stock_Snapshot_Date"), errors="coerce").dt.date
    work["Earnings_Date"] = pick("Earnings_Date", "earnings_date")
    work["Option_Expiry_Date"] = pick("Option_Expiry_Date", "Call_Option_Expiry_Date", "Put_Option_Expiry_Date", "calls_exDate", "calls_ExDate")
    work["Average_Volume"] = pd.to_numeric(pick("Average_Volume", "average_volume"), errors="coerce")
    work["Volume"] = pd.to_numeric(pick("Volume", "volume"), errors="coerce")
    work["calls_OpenInterest"] = pd.to_numeric(
        pick("calls_OpenInterest", "calls_openInterest", "Call_OpenInterest"),
        errors="coerce",
    )
    work["Beta"] = pd.to_numeric(pick("Beta", "beta"), errors="coerce")
    work["Market_Cap"] = pd.to_numeric(work.get("Market_Cap"), errors="coerce")
    work["Close_Price"] = pd.to_numeric(work.get("Stock_Price"), errors="coerce")
    work["options_price"] = np.where(
        work.get("chosen_action").eq("call"),
        pd.to_numeric(work.get("Call_Option_Price"), errors="coerce"),
        pd.to_numeric(work.get("Put_Option_Price"), errors="coerce"),
    )
    work["price_gap"] = pd.to_numeric(work.get("strike_to_close_price_gap"), errors="coerce")
    work["calls_strike"] = pd.to_numeric(work.get("Call_Option_Strike"), errors="coerce")
    work["days_to_earnings"] = pd.to_numeric(work.get("days_to_earnings"), errors="coerce")
    # Keep as numeric to avoid nullable-int dtype issues across pandas versions.
    work["WinorLoss"] = pd.to_numeric(work.get("win_loss_flag"), errors="coerce")
    work["evaluation_status"] = work.get("evaluation_status")
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
    work["eod_nextday_High"] = pd.to_numeric(work.get("joined_eod_nextday_High"), errors="coerce")
    work["eod_nextday_Low"] = pd.to_numeric(work.get("joined_eod_nextday_Low"), errors="coerce")

    # Fill required-but-possibly-missing fields as NULL.
    for col in FINAL_OPT_SCHEMA_COLS:
        if col not in work.columns:
            work[col] = pd.NA

    return work[FINAL_OPT_SCHEMA_COLS].copy()


def run_optimization_pipeline(
    source_df: pd.DataFrame,
    threshold: float,
    budget: float,
    max_investment_fraction: float,
    max_portfolios: int,
    evaluated_only: bool,
) -> pd.DataFrame:
    # End-to-end post-selection stage:
    # enrich -> filter by constraints -> optimize per date -> schema align.
    enriched = source_df.copy()
    if evaluated_only and "evaluation_status" in enriched.columns:
        enriched = enriched[enriched["evaluation_status"] == "EVALUATED"].copy()

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

    # Liquidity filters (p05 Beat thresholds — best discrimination ratio while retaining 95% of Beats)
    # Volume dropped: discrimination ratio < 1 at p10, not a useful filter
    enriched["Relative_Volume"] = pd.to_numeric(enriched.get("Relative_Volume"), errors="coerce")
    enriched["Market_Cap"] = pd.to_numeric(enriched.get("Market_Cap"), errors="coerce")
    enriched = enriched[enriched["Relative_Volume"] >= 0.73].copy()
    enriched = enriched[enriched["Market_Cap"] >= 80.82].copy()

    enriched["expected_return_based_on_prob"] = enriched["options_price"]
    enriched["Threshold"] = float(threshold)
    enriched["Budget"] = int(budget)
    enriched["Max_Investment_Fraction"] = float(max_investment_fraction)
    enriched["Max_Portfolios"] = int(max_portfolios)

    selected_parts = []
    for date, group in enriched.groupby("Stock_Snapshot_Date", dropna=False):
        days_col = pd.to_numeric(group.get("days_to_earnings"), errors="coerce")

        # Split into two groups: earnings day = 1 and the rest.
        earnings_group = group[days_col == 1].copy()
        rest_group = group[days_col != 1].copy()

        # Each group gets half the budget; if no earnings stocks exist, rest gets the full budget.
        half = budget * 0.5
        has_earnings = not earnings_group.empty

        earnings_result = solve_knapsack(earnings_group, half, max_investment_fraction, max_portfolios) if has_earnings else pd.DataFrame()
        earnings_spent = earnings_result["Investment"].sum() if not earnings_result.empty else 0
        rest_budget = (half + (half - earnings_spent)) if has_earnings else budget
        rest_slots = max_portfolios - len(earnings_result)
        rest_result = solve_knapsack(rest_group, rest_budget, max_investment_fraction, rest_slots) if not rest_group.empty and rest_slots > 0 else pd.DataFrame()

        # Union both results for this date.
        date_result = pd.concat([earnings_result, rest_result], ignore_index=True)
        if not date_result.empty:
            selected_parts.append(date_result)

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
    all_preds = normalize_market_cap_column(all_preds)

    # Stage 2: choose one best action per ticker/date.
    ensure_required_columns(all_preds)
    selected = select_best_action(all_preds, args.score_column)
    write_output(selected, args.output_csv, args.output_bq_table, args.write_disposition)

    # Stage 3 (optional): run constrained optimization and write FOR_EXTERNAL table.
    if args.run_optimization:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed; cannot run optimization pipeline")
        if not args.output_bq_table:
            raise ValueError("--output-bq-table is required with --run-optimization so SQL can read persisted best-action rows")

        # Resolve optimization output destinations:
        # - explicit live output arg for paper/live trading
        # - backward-compatible single table arg maps to evaluated output
        out_live = args.output_optimization_live_bq_table
        out_eval = args.output_optimization_bq_table
        if not out_live and not out_eval:
            raise ValueError(
                "Provide at least one optimization output table: "
                "--output-optimization-live-bq-table and/or --output-optimization-bq-table."
            )

        client = bigquery.Client()
        evaluated = add_winloss_from_nextday(
            client=client,
            best_action_table=args.output_bq_table,
            nextday_table=args.nextday_bq_table,
        )
        if args.output_evaluated_bq_table:
            write_output(evaluated, "", args.output_evaluated_bq_table, args.write_disposition)

        # Optimization output table: use evaluated source so evaluation_status is preserved
        # (EVALUATED / PENDING_NEXT_DAY_DATA based on next-day availability).
        if out_live:
            live_optimized = run_optimization_pipeline(
                source_df=evaluated,
                threshold=args.threshold,
                budget=args.budget,
                max_investment_fraction=args.max_investment_fraction,
                max_portfolios=args.max_portfolios,
                evaluated_only=False,
            )
            write_output(live_optimized, "", out_live, args.write_disposition)

        # EVALUATED optimization: only rows with next-day market data available.
        if out_eval:
            evaluated_optimized = run_optimization_pipeline(
                source_df=evaluated,
                threshold=args.threshold,
                budget=args.budget,
                max_investment_fraction=args.max_investment_fraction,
                max_portfolios=args.max_portfolios,
                evaluated_only=True,
            )
            write_output(evaluated_optimized, "", out_eval, args.write_disposition)


if __name__ == "__main__":
    main()
