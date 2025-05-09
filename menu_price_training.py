from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

RANDOM_STATE = 42


def load_sample(csv_path: Path, n_rows: int | None = None) -> pd.DataFrame:
    print(f"Loading {n_rows or 'all'} rows from {csv_path} …")
    df = pd.read_csv(csv_path, nrows=n_rows)

    # Basic cleaning
    df.dropna(subset=["price", "name"], inplace=True)
    df["name"] = df["name"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()

    # Clean price strings like "$12.99 USD"
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
    )
    df.dropna(subset=["price"], inplace=True)
    return df


def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=100_000,
        lowercase=True,
    )

    lgbm = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("tfidf", tfidf),
        ("lgbm", lgbm),
    ])
    return pipe


def train(df: pd.DataFrame, output_path: Path):
    X = df["name"]
    y = np.log1p(df["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()
    print("Training LightGBM pipeline …")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(preds)))
    print(f"RMSE on hold‑out set: {rmse:.2f}")

    joblib.dump(pipe, output_path)
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train menu‑item price model")
    parser.add_argument("--csv", type=str, default="restaurant-menus_cleaned.csv", help="Path to cleaned menu CSV")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to sample (for RAM)")
    parser.add_argument("--output", type=str, default="lgbm_menu_price.pkl")
    args = parser.parse_args()

    df = load_sample(Path(args.csv), n_rows=args.sample)
    train(df, Path(args.output))


if __name__ == "__main__":
    main()
