import argparse

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation/test splits in BigQuery")
    parser.add_argument("--source-bq-table", required=True, help="Source table project.dataset.table")
    parser.add_argument("--output-bq-table", required=True, help="Output split table project.dataset.table")
    parser.add_argument("--project-id", default="", help="Optional project override")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="Seed used in hash key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs((args.train_ratio + args.validation_ratio + args.test_ratio) - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery not installed")

    train_cutoff = int(args.train_ratio * 10000)
    val_cutoff = int((args.train_ratio + args.validation_ratio) * 10000)

    query = f"""
    CREATE OR REPLACE TABLE `{args.output_bq_table}` AS
    WITH base AS (
      SELECT
        t.*,
        ABS(MOD(FARM_FINGERPRINT(CONCAT(CAST({args.seed} AS STRING), '::', COALESCE(CAST(Ticker AS STRING), ''), '::', COALESCE(CAST(Stock_Snapshot_Date AS STRING), ''))), 10000)) AS split_bucket
      FROM `{args.source_bq_table}` t
    )
    SELECT
      * EXCEPT(split_bucket),
      CASE
        WHEN split_bucket < {train_cutoff} THEN 'train'
        WHEN split_bucket < {val_cutoff} THEN 'validation'
        ELSE 'test'
      END AS split
    FROM base
    """

    client = bigquery.Client(project=args.project_id or None)
    client.query(query).result()
    print(f"Created split table: {args.output_bq_table}")
    print(f"Ratios train/validation/test: {args.train_ratio}/{args.validation_ratio}/{args.test_ratio}")


if __name__ == "__main__":
    main()
