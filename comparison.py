# weight_sweep.py
"""Grid‑search the pseudo‑label weight (0.0 → 1.0) for the self‑training
restaurant‑score model and plot RMSE vs weight.

Usage:
    python weight_sweep.py --data restaurants_cleaned.csv

Requirements:
    pip install pandas numpy catboost scikit-learn matplotlib joblib

The script will:
1. Load the already‑cleaned CSV that has `score`, `score_missing`,
   `price_range`, `price_range_missing`, `ratings`, `ratings_missing`,
   `category`.
2. Train a first‑pass CatBoost on labeled rows only.
3. For each pseudo‑label weight w ∈ {0.0, 0.05, …, 1.0}:
       * Retrain on the full dataset with that weight.
       * Record RMSE on the labeled validation set.
4. Save a PNG plot `rmse_vs_weight.png` and print a table of scores.

Tip: If you need finer resolution, change `WEIGHTS = np.linspace(...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

RANDOM_STATE = 42
WEIGHTS = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, …, 1.0

CAT_COLS = ["category"]
NUM_COLS = [
    "price_range", "price_range_missing",
    "ratings", "ratings_missing"
]
FEAT_COLS = CAT_COLS + NUM_COLS


def train_first_pass(df: pd.DataFrame):
    """Train on labeled rows only and return the model & validation split."""
    labeled_df = df[df["score_missing"] == 0].copy()

    X = labeled_df[FEAT_COLS]
    y = labeled_df["score"]

    # Stratify on category but collapse singletons
    vc = labeled_df["category"].value_counts()
    strat_col = labeled_df["category"].where(~labeled_df["category"].isin(vc[vc < 2].index), "other")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat_col
    )

    train_pool = Pool(X_train, y_train, cat_features=CAT_COLS)
    val_pool = Pool(X_val, y_val, cat_features=CAT_COLS)

    model = CatBoostRegressor(
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    val_pred = model.predict(val_pool)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    return model, rmse, (X_val, y_val)


def self_train(df: pd.DataFrame, base_model: CatBoostRegressor, weight: float):
    """Return RMSE on labeled validation set after self‑training with given weight."""
    mask_unlabeled = df["score_missing"] == 1

    # Generate pseudo‑labels once per weight sweep to save time
    pseudos = base_model.predict(Pool(df.loc[mask_unlabeled, FEAT_COLS], cat_features=CAT_COLS))

    df["y_final"] = df["score"]
    df.loc[mask_unlabeled, "y_final"] = pseudos 
    sample_weight = np.where(mask_unlabeled, weight, 1.0)

    final_pool = Pool(df[FEAT_COLS], df["y_final"], cat_features=CAT_COLS, weight=sample_weight)

    final_model = CatBoostRegressor(
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50,
    )

    final_model.fit(final_pool, use_best_model=False)

    return final_model


def main(path: str):
    df = pd.read_csv(path)

    print("Training first‑pass model on labeled rows …")
    base_model, base_rmse, val_split = train_first_pass(df)
    print(f"Pass‑1 RMSE: {base_rmse:.4f}\n")

    X_val, y_val = val_split
    val_pool_base = Pool(X_val, y_val, cat_features=CAT_COLS)

    rmses = []
    for w in WEIGHTS:
        print(f"Self‑training with pseudo weight = {w:.2f} …", end=" ")
        model_w = self_train(df.copy(), base_model, weight=w)
        preds = model_w.predict(val_pool_base)
        rmse_w = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse_w)
        print(f"RMSE = {rmse_w:.4f}")

    # ------------------------------------------------------------------
    # Plot RMSE vs weight
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(WEIGHTS, rmses, marker="o")
    plt.xlabel("Pseudo‑label weight")
    plt.ylabel("RMSE on labeled validation set")
    plt.title("Self‑training weight sweep")
    plt.grid(True, alpha=0.3)
    plt.savefig("rmse_vs_weight.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary table
    print("\nWeight  RMSE")
    for w, r in zip(WEIGHTS, rmses):
        print(f"{w:5.2f}  {r:.4f}")

    # Persist results for further analysis
    results = pd.DataFrame({"weight": WEIGHTS, "rmse": rmses})
    results.to_csv("rmse_vs_weight.csv", index=False)
    joblib.dump(results, "rmse_vs_weight.pkl")

    print("\nArtifacts saved: rmse_vs_weight.png, rmse_vs_weight.csv, rmse_vs_weight.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep pseudo‑label weights and plot RMSE")
    parser.add_argument("--data", type=str, default="restaurants_cleaned.csv", help="Path to cleaned restaurant CSV")
    args = parser.parse_args()
    main(args.data)
