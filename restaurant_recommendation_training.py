from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

EMBED_DIM = 100
WINDOW = 5
MIN_COUNT = 2
WORKERS = 4


def tokenize(name: str) -> list[str]:
    return name.lower().strip().replace(" ", "_").split()


def build_corpus(menus_path: Path) -> tuple[list[list[str]], dict[int, list[str]]]:
    menus = pd.read_csv(menus_path, usecols=["restaurant_id", "name"])
    menus.dropna(subset=["name"], inplace=True)
    menus["tokens"] = menus["name"].astype(str).apply(tokenize)

    corpus = []                     # list of token lists per restaurant
    rest_to_items: dict[int, list[str]] = {}

    for rid, grp in menus.groupby("restaurant_id"):
        tokens = [tok for sublist in grp["tokens"] for tok in sublist]
        if len(tokens) >= MIN_COUNT:
            corpus.append(tokens)
            rest_to_items[rid] = tokens
    return corpus, rest_to_items


def train_word2vec(corpus: list[list[str]]) -> Word2Vec:
    print("Training Word2Vec on", len(corpus), "restaurants …")
    model = Word2Vec(
        sentences=corpus,
        vector_size=EMBED_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,
        negative=10,
        epochs=10,
    )
    return model


def compute_restaurant_vectors(model: Word2Vec, rest_items: dict[int, list[str]]):
    vectors = []
    rest_ids = []
    for rid, tokens in rest_items.items():
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
            rest_ids.append(rid)
    return np.vstack(vectors).astype(np.float32), rest_ids


def main():
    p = argparse.ArgumentParser(description="Train restaurant recommender via Item2Vec")
    p.add_argument("--menus", default="restaurant-menus_cleaned.csv", help="Path to cleaned menu CSV")
    p.add_argument("--restaurants", default="restaurants_cleaned.csv", help="Path to cleaned restaurants CSV")
    p.add_argument("--output_dir", default=".")
    args = p.parse_args()

    corpus, rest_items = build_corpus(Path(args.menus))
    model = train_word2vec(corpus)

    model.save("item2vec.model")

    rest_vecs, rest_ids = compute_restaurant_vectors(model, rest_items)
    np.save("restaurant_vectors.npy", rest_vecs)

    # also save a mapping to restaurant name for easy lookup
    rest_df = pd.read_csv(args.restaurants, usecols=["id", "name"])
    mapping = {
        idx: {"restaurant_id": int(rid), "name": rest_df.loc[rest_df["id"] == rid, "name"].values[0]}
        for idx, rid in enumerate(rest_ids)
    }
    with open("restaurant_index.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" •", "item2vec.model")
    print(" •", "restaurant_vectors.npy")
    print(" •", "restaurant_index.json")
    print("Done.")


if __name__ == "__main__":
    main()
