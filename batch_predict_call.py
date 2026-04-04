import argparse
import io
import os
from typing import Optional

import joblib
import pandas as pd

try:
    from google.cloud import storage
except Exception:
    storage = None

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None


MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model_call.joblib")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "models/label_encoder_call.joblib")
PREDICTION_FLAG_TYPE = "call"
POSITION_VALUE_COL = "total_investment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prediction runner (call model)")
    parser.add_argument("--input", default="", help="Input CSV path, local or gs://bucket/path.csv")
    parser.add_argument("--input-bq-table", default="", help="Input BigQuery table in project.dataset.table format")
    parser.add_argument("--input-bq-split", default="", help="Optional split filter value (e.g. test)")
    parser.add_argument("--input-bq-year", type=int, default=2026, help="Optional year filter for Stock_Snapshot_Date")
    parser.add_argument("--input-bq-date", default="", help="Optional exact Stock_Snapshot_Date filter in YYYY-MM-DD")
    parser.add_argument("--output", default="", help="Output CSV path, local or gs://bucket/path.csv")
    parser.add_argument("--chunksize", type=int, default=5000, help="Rows per prediction chunk")
    parser.add_argument("--return-proba", action="store_true", help="Include probability columns when model supports it")
    parser.add_argument(
        "--output-bq-table",
        default="",
        help="Optional BigQuery table in project.dataset.table format. If set, writes predictions to BigQuery instead of CSV output.",
    )
    parser.add_argument(
        "--id-columns",
        default="",
        help="Comma-separated identifier columns to keep and place first in output (e.g. Ticker,Stock_Snapshot_Date)",
    )
    return parser.parse_args()


def is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def split_gcs_uri(uri: str) -> tuple[str, str]:
    if not is_gcs(uri):
        raise ValueError(f"Not a GCS URI: {uri}")
    no_scheme = uri[len("gs://") :]
    parts = no_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return parts[0], parts[1]


def read_csv_any(path: str) -> pd.DataFrame:
    if not is_gcs(path):
        return pd.read_csv(path)

    if storage is None:
        raise RuntimeError("google-cloud-storage not installed; cannot read gs:// paths")

    bucket_name, blob_name = split_gcs_uri(path)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def read_bq_any(table: str, split_value: str, year_value: int, date_value: str) -> pd.DataFrame:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery not installed; cannot read BigQuery input")
    client = bigquery.Client()
    where_clauses = []
    params = []
    if split_value:
        where_clauses.append("split = @split")
        params.append(bigquery.ScalarQueryParameter("split", "STRING", split_value))
    if date_value:
        where_clauses.append("DATE(Stock_Snapshot_Date) = @snapshot_date")
        params.append(bigquery.ScalarQueryParameter("snapshot_date", "DATE", date_value))
    if year_value:
        where_clauses.append("EXTRACT(YEAR FROM DATE(Stock_Snapshot_Date)) = @year")
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year_value))
    query = f"SELECT * FROM `{table}`"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
    return client.query(query, job_config=job_config).to_dataframe()


def write_csv_any(df: pd.DataFrame, path: str) -> None:
    if not is_gcs(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return

    if storage is None:
        raise RuntimeError("google-cloud-storage not installed; cannot write gs:// paths")

    bucket_name, blob_name = split_gcs_uri(path)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")


def load_optional_encoder(path: str) -> Optional[object]:
    if os.path.exists(path):
        return joblib.load(path)
    return None


def prepare_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Stock_Snapshot_Date" in out.columns:
        out["Stock_Snapshot_Date"] = pd.to_datetime(out["Stock_Snapshot_Date"], errors="coerce").dt.date
    if "next_day_Snapshot_Date" in out.columns:
        out["next_day_Snapshot_Date"] = pd.to_datetime(out["next_day_Snapshot_Date"], errors="coerce").dt.date
    float_cols = [
        "Stock_Price",
        "Performance_Week",
        "Average_True_Range",
        "Change",
        "Call_Option_Strike",
        "Call_Option_Price",
        "Put_Option_Price",
        "strike_to_close_price_gap",
        "todays_range",
        "prob_Beat",
        "prob_NoBeat",
        POSITION_VALUE_COL,
        "eod_nextday_High",
        "eod_nextday_Low",
    ]
    int_cols = ["days_to_earnings", "days_earnings_to_expiry", "days_to_options_expiry"]

    for col in float_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in int_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def align_to_bq_table_columns(client: "bigquery.Client", table_id: str, df: pd.DataFrame) -> pd.DataFrame:
    table = client.get_table(table_id)
    table_cols = [f.name for f in table.schema]
    table_col_set = set(table_cols)
    keep_cols = [c for c in table_cols if c in df.columns]
    dropped_cols = [c for c in df.columns if c not in table_col_set]
    if dropped_cols:
        print(f"Dropping columns not present in destination table {table_id}: {dropped_cols}")
    if not keep_cols:
        raise ValueError(f"No overlapping columns between result and destination table schema: {table_id}")
    return df[keep_cols].copy()


def add_position_value(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["Stock_Price", "Call_Option_Price"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"input is missing required columns for {POSITION_VALUE_COL}: {missing}")
    out[POSITION_VALUE_COL] = (pd.to_numeric(out["Stock_Price"], errors="coerce") * 100) + (
        pd.to_numeric(out["Call_Option_Price"], errors="coerce") * 100
    )
    return out


def ensure_nextday_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "eod_nextday_Low" not in out.columns:
        if "nextday_Low" in out.columns:
            out["eod_nextday_Low"] = out["nextday_Low"]
        else:
            out["eod_nextday_Low"] = pd.NA
    if "eod_nextday_High" not in out.columns:
        if "eod_next_High" in out.columns:
            out["eod_nextday_High"] = out["eod_next_High"]
        elif "nextday_High" in out.columns:
            out["eod_nextday_High"] = out["nextday_High"]
        else:
            out["eod_nextday_High"] = pd.NA
    return out


def get_feature_frame(df: pd.DataFrame, model: object) -> pd.DataFrame:
    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if not model_feature_names:
        return df
    missing_features = [c for c in model_feature_names if c not in df.columns]
    if missing_features:
        raise ValueError(f"input is missing required model feature columns: {missing_features}")
    return df[model_feature_names]


def main() -> None:
    args = parse_args()
    if bool(args.input) == bool(args.input_bq_table):
        raise ValueError("Provide exactly one input source: --input OR --input-bq-table")
    if not args.output and not args.output_bq_table:
        raise ValueError("provide at least one output target: --output or --output-bq-table")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    label_encoder = load_optional_encoder(LABEL_ENCODER_PATH)

    if args.input:
        df = read_csv_any(args.input)
    else:
        df = read_bq_any(args.input_bq_table, args.input_bq_split, args.input_bq_year, args.input_bq_date)
    if df.empty:
        if args.output:
            write_csv_any(pd.DataFrame(columns=["prediction"]), args.output)
            print("Input empty. Wrote empty predictions file.")
        else:
            print(f"Input empty. No rows to write to BigQuery table: {args.output_bq_table}")
        return

    feature_df = get_feature_frame(df, model)

    id_columns = [c.strip() for c in args.id_columns.split(",") if c.strip()]
    missing_ids = [c for c in id_columns if c not in df.columns]
    if missing_ids:
        raise ValueError(f"id columns not found in input: {missing_ids}")

    parts = []
    for start in range(0, len(df), args.chunksize):
        chunk = df.iloc[start : start + args.chunksize]
        feature_chunk = feature_df.iloc[start : start + args.chunksize]

        preds = model.predict(feature_chunk)
        out_chunk = chunk.copy()
        out_chunk["prediction_flag_type"] = PREDICTION_FLAG_TYPE

        if label_encoder is not None:
            try:
                out_chunk["prediction"] = label_encoder.inverse_transform(preds)
            except Exception:
                out_chunk["prediction"] = preds
        else:
            out_chunk["prediction"] = preds

        if args.return_proba and hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_chunk)
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                proba_cols = [f"prob_{c}" for c in label_encoder.classes_]
            else:
                proba_cols = [f"prob_{i}" for i in range(probs.shape[1])]
            proba_df = pd.DataFrame(probs, columns=proba_cols, index=chunk.index)
            out_chunk = pd.concat([out_chunk, proba_df], axis=1)

        if id_columns:
            first_cols = id_columns + ["prediction_flag_type"]
            first_cols = [c for c in first_cols if c in out_chunk.columns]
            remaining_cols = [c for c in out_chunk.columns if c not in first_cols]
            out_chunk = out_chunk[first_cols + remaining_cols]

        parts.append(out_chunk)

    result = pd.concat(parts, axis=0)
    result = ensure_nextday_columns(result)
    result = add_position_value(result)

    if args.output_bq_table:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed; cannot write BigQuery output")
        result = prepare_for_bigquery(result)
        client = bigquery.Client()
        result = align_to_bq_table_columns(client, args.output_bq_table, result)
        load_job = client.load_table_from_dataframe(
            result,
            args.output_bq_table,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        )
        load_job.result()
        print(f"Batch prediction complete. Rows: {len(result)} | BigQuery table: {args.output_bq_table}")
    else:
        write_csv_any(result, args.output)
        print(f"Batch prediction complete. Rows: {len(result)} | Output: {args.output}")


if __name__ == "__main__":
    main()
