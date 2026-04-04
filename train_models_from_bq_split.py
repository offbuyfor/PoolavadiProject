import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None


ID_COLS = ["Ticker", "Stock_Snapshot_Date"]
CALL_TARGET = "price_value_delta_above_call_flag"
PUT_TARGET = "price_value_delta_above_put_flag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train call and put models from BigQuery split table")
    parser.add_argument("--input-bq-table", required=True, help="Input table in project.dataset.table format with split column")
    parser.add_argument("--split-col", default="split", help="Column name containing split labels")
    parser.add_argument("--train-split", default="train", help="Split label for training data")
    parser.add_argument("--validation-split", default="validation", help="Split label for validation data")
    parser.add_argument("--test-split", default="test", help="Split label for test data")
    parser.add_argument("--out-dir", default="models", help="Directory to save trained artifacts")
    parser.add_argument("--project-id", default="", help="Optional GCP project override")
    return parser.parse_args()


def read_split_table(table: str, project_id: str) -> pd.DataFrame:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery not installed")
    client = bigquery.Client(project=project_id or None)
    query = f"SELECT * FROM `{table}`"
    return client.query(query).to_dataframe()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_one_target(df: pd.DataFrame, split_col: str, train_label: str, val_label: str, test_label: str, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if split_col not in df.columns:
        raise ValueError(f"Split column not found: {split_col}")

    keep_df = df[df[target_col].notna()].copy()
    drop_cols = [CALL_TARGET, PUT_TARGET, split_col] + [c for c in ID_COLS if c in keep_df.columns]
    feature_cols = [c for c in keep_df.columns if c not in drop_cols]

    train_df = keep_df[keep_df[split_col] == train_label].copy()
    val_df = keep_df[keep_df[split_col] == val_label].copy()
    test_df = keep_df[keep_df[split_col] == test_label].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more required splits are empty. Check split labels and split table content.")

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    preprocessor = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train_enc)

    val_pred = le.inverse_transform(model.predict(X_val))
    test_pred = le.inverse_transform(model.predict(X_test))

    print(f"\n=== {target_col} ===")
    print("Validation accuracy:", accuracy_score(y_val, val_pred))
    print("Validation report:\n", classification_report(y_val, val_pred))
    print("Test accuracy:", accuracy_score(y_test, test_pred))
    print("Test report:\n", classification_report(y_test, test_pred))

    return model, le


def main() -> None:
    args = parse_args()
    df = read_split_table(args.input_bq_table, args.project_id)

    call_model, call_le = train_one_target(
        df, args.split_col, args.train_split, args.validation_split, args.test_split, CALL_TARGET
    )
    put_model, put_le = train_one_target(
        df, args.split_col, args.train_split, args.validation_split, args.test_split, PUT_TARGET
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(call_model, out_dir / "best_model_call.joblib")
    joblib.dump(call_le, out_dir / "label_encoder_call.joblib")
    joblib.dump(put_model, out_dir / "best_model_put.joblib")
    joblib.dump(put_le, out_dir / "label_encoder_put.joblib")

    print("\nSaved models:")
    print(out_dir / "best_model_call.joblib")
    print(out_dir / "label_encoder_call.joblib")
    print(out_dir / "best_model_put.joblib")
    print(out_dir / "label_encoder_put.joblib")


if __name__ == "__main__":
    main()
