from __future__ import annotations

import json

import numpy as np
from numpy.linalg import norm

vecs = np.load("restaurant_vectors.npy")

with open("restaurant_index.json", "r", encoding="utf-8") as f:
    idx_map = json.load(f)

names = [idx_map[str(i)]["name"] for i in range(len(idx_map))]

query = "bunrise burgers"

q_lower = query.lower()
matches = [i for i, n in enumerate(names) if q_lower in n.lower()]
if not matches:
    raise ValueError(f"No restaurant found containing: {query}")
q_idx = matches[0]

q_vec = vecs[q_idx]

cos_sim = vecs @ q_vec / (norm(vecs, axis=1) * norm(q_vec) + 1e-9)
ranked = np.argsort(-cos_sim)

print(f"\nQuery: {names[q_idx]}\n")
print("Top", 3, "similar restaurants:")
count = 0
for idx in ranked:
    if idx == q_idx:
        continue
    print(f"  {names[idx]}   (cos sim = {cos_sim[idx]:.3f})")
    count += 1
    if count >= 3:
        break
