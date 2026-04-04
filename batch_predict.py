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
PUT_MODEL_PATH = os.environ.get("PUT_MODEL_PATH", "")
PUT_LABEL_ENCODER_PATH = os.environ.get("PUT_LABEL_ENCODER_PATH", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prediction runner")
    parser.add_argument("--input", required=True, help="Input CSV path, local or gs://bucket/path.csv")
    parser.add_argument("--output", default="", help="Output CSV path, local or gs://bucket/path.csv")
    parser.add_argument(
        "--put-model-path",
        default=PUT_MODEL_PATH,
        help="Optional model path for put-flag predictions. If provided, output includes both call and put predictions.",
    )
    parser.add_argument(
        "--put-label-encoder-path",
        default=PUT_LABEL_ENCODER_PATH,
        help="Optional label encoder path for put model. Defaults to call label encoder when omitted.",
    )
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
    return out


def get_feature_frame(df: pd.DataFrame, model: object) -> pd.DataFrame:
    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if not model_feature_names:
        return df
    missing_features = [c for c in model_feature_names if c not in df.columns]
    if missing_features:
        raise ValueError(f"input is missing required model feature columns: {missing_features}")
    return df[model_feature_names]


def predict_chunk(
    base_chunk: pd.DataFrame,
    feature_chunk: pd.DataFrame,
    model: object,
    label_encoder: Optional[object],
    return_proba: bool,
    flag_type: str,
) -> pd.DataFrame:
    preds = model.predict(feature_chunk)
    out_chunk = base_chunk.copy()
    out_chunk["prediction_flag_type"] = flag_type

    if label_encoder is not None:
        try:
            out_chunk["prediction"] = label_encoder.inverse_transform(preds)
        except Exception:
            out_chunk["prediction"] = preds
    else:
        out_chunk["prediction"] = preds

    if return_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_chunk)
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            proba_cols = [f"prob_{c}" for c in label_encoder.classes_]
        else:
            proba_cols = [f"prob_{i}" for i in range(probs.shape[1])]
        proba_df = pd.DataFrame(probs, columns=proba_cols, index=base_chunk.index)
        out_chunk = pd.concat([out_chunk, proba_df], axis=1)

    return out_chunk


def main() -> None:
    args = parse_args()
    if not args.output and not args.output_bq_table:
        raise ValueError("provide at least one output target: --output or --output-bq-table")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    label_encoder = load_optional_encoder(LABEL_ENCODER_PATH)
    put_model = None
    put_label_encoder = None
    if args.put_model_path:
        if not os.path.exists(args.put_model_path):
            raise FileNotFoundError(f"Put model file not found: {args.put_model_path}")
        put_model = joblib.load(args.put_model_path)
        put_label_encoder = load_optional_encoder(args.put_label_encoder_path) if args.put_label_encoder_path else label_encoder

    df = read_csv_any(args.input)
    if df.empty:
        write_csv_any(pd.DataFrame(columns=["prediction"]), args.output)
        print("Input empty. Wrote empty predictions file.")
        return

    feature_df_call = get_feature_frame(df, model)
    feature_df_put = get_feature_frame(df, put_model) if put_model is not None else None

    id_columns = [c.strip() for c in args.id_columns.split(",") if c.strip()]
    missing_ids = [c for c in id_columns if c not in df.columns]
    if missing_ids:
        raise ValueError(f"id columns not found in input: {missing_ids}")

    parts = []
    for start in range(0, len(df), args.chunksize):
        chunk = df.iloc[start : start + args.chunksize]
        feature_chunk_call = feature_df_call.iloc[start : start + args.chunksize]
        call_chunk = predict_chunk(
            base_chunk=chunk,
            feature_chunk=feature_chunk_call,
            model=model,
            label_encoder=label_encoder,
            return_proba=args.return_proba,
            flag_type="call",
        )
        chunks_out = [call_chunk]

        if put_model is not None and feature_df_put is not None:
            feature_chunk_put = feature_df_put.iloc[start : start + args.chunksize]
            put_chunk = predict_chunk(
                base_chunk=chunk,
                feature_chunk=feature_chunk_put,
                model=put_model,
                label_encoder=put_label_encoder,
                return_proba=args.return_proba,
                flag_type="put",
            )
            chunks_out.append(put_chunk)

        for out_chunk in chunks_out:
            if id_columns:
                first_cols = id_columns + ["prediction_flag_type"]
                first_cols = [c for c in first_cols if c in out_chunk.columns]
                remaining_cols = [c for c in out_chunk.columns if c not in first_cols]
                out_chunk = out_chunk[first_cols + remaining_cols]
            parts.append(out_chunk)

    result = pd.concat(parts, axis=0)

    if args.output_bq_table:
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not installed; cannot write BigQuery output")
        result = prepare_for_bigquery(result)
        client = bigquery.Client()
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
