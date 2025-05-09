import pandas as pd
from catboost import CatBoostRegressor, Pool

model = CatBoostRegressor()
model.load_model("catboost_restaurant_score_final.cbm")

custom_row = pd.DataFrame({
    "category": ["indian"],

    "price_range": [3],
    "price_range_missing": [0],

    "ratings": [120],
    "ratings_missing": [0]
})

cat_cols = ["category"]
pool = Pool(custom_row, cat_features=cat_cols)

predicted_score = model.predict(pool)[0]
print(f"Predicted rating: {predicted_score:.2f}")