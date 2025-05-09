from __future__ import annotations

import numpy as np
import joblib
import pandas as pd

pipe = joblib.load("lgbm_menu_price.pkl")

examples = [
    "truffle mac and cheese",
    "spicy paneer wrap",
    "Cheese Garlic Naan"
]
item_names = examples

X_custom = pd.Series(item_names)

log_preds = pipe.predict(X_custom)
price_preds = pd.Series(np.expm1(log_preds)).round(2)

for dish, price in zip(item_names, price_preds):
    print(f"{dish} ➜ ${price}")
