# Eatsight

This project explores machine learning approaches for predicting restaurant ratings and menu item prices, as well as building a simple restaurant recommendation system, using the [Kaggle - Uber Eats USA Restaurants Menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus) dataset.

## 🧠 Project Structure

- **`pre-processing.ipynb`**: Cleans and preprocesses the raw restaurant and menu datasets, encodes features, and handles missing values.
- **`rating_prediction_training.ipynb`**: Trains a CatBoost regression model to predict restaurant ratings, applies pseudo-labeling for self-training, and evaluates the effect of pseudo-label weights.
- **`comparison.py`**: Sweeps across different pseudo-label weights (0.0 to 1.0), evaluating RMSE on the validation set and visualizing the effect of weight tuning.
- **`rating_prediction.py`**: Loads the trained rating model and predicts the rating for a custom restaurant input.
- **`menu_price_training.py`**: Trains a LightGBM model to predict menu item prices from item names using TF-IDF features.
- **`menu_price_prediction.py`**: Loads the menu price model and predicts prices for custom menu items.
- **`restaurant_recommendation_training.py`**: Trains a Word2Vec (Item2Vec) model on menu item names to generate restaurant embeddings for similarity-based recommendations.
- **`restaurant_recommendation_prediction.py`**: Finds and prints the most similar restaurants to a given query using the trained embeddings.

## 📁 Dataset

**Source**: [Kaggle - Uber Eats USA Restaurants Menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus)

After preprocessing, each restaurant entry includes:

- `score`: Numerical rating (target variable, partially missing).
- `price_range`: Numerical encoding of price tier.
- `ratings`: Aggregated user ratings.
- `category`: Type of cuisine.
- `*_missing`: Binary flags for missingness of target or features.

## ⚙️ Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Preprocessing

Clean and prepare the data:

```bash
jupyter notebook pre-processing.ipynb
```

This will generate `restaurants_cleaned.csv` and `restaurant-menus_cleaned.csv`.

### 2. Restaurant Rating Prediction

**a. Pseudo-label Weight Sweep**

Run the grid search over pseudo-label weights:

```bash
python comparison.py --data restaurants_cleaned.csv
```

Artifacts generated:
- `rmse_vs_weight.png`: RMSE vs. pseudo-label weight plot.
- `rmse_vs_weight.csv`: Numerical results.
- `rmse_vs_weight.pkl`: Serialized results.

**b. Train and Evaluate Final Model**

Train the baseline and self-training models:

```bash
jupyter notebook rating_prediction_training.ipynb
```

**c. Predict a Restaurant's Rating**

Edit and run:

```bash
python rating_prediction.py
```

## 🍽️ Menu Item Price Prediction

**a. Train the Model**

```bash
python menu_price_training.py --csv restaurant-menus_cleaned.csv --output lgbm_menu_price.pkl
```

**b. Predict Menu Item Prices**

Edit and run:

```bash
python menu_price_prediction.py
```

## 🏆 Restaurant Recommendation (Similarity)

**a. Train Embeddings**

```bash
python restaurant_recommendation_training.py --menus restaurant-menus_cleaned.csv --restaurants restaurants_cleaned.csv
```

**b. Find Similar Restaurants**

Edit the query in:

```bash
python restaurant_recommendation_prediction.py
```

## 📈 Goal

Evaluate the effect of incorporating pseudo-labeled data on model performance, predict menu prices from item names, and recommend similar restaurants using menu-based embeddings.

---
**Artifacts produced:**  
- Cleaned datasets: `restaurants_cleaned.csv`, `restaurant-menus_cleaned.csv`  
- Trained models: `catboost_restaurant_score_final.cbm`, `lgbm_menu_price.pkl`, `item2vec.model`  
- Evaluation plots and tables: `rmse_vs_weight.png`, `rmse_vs_weight.csv`  
- Embeddings and mappings: `restaurant_vectors.npy`, `restaurant_index.json`
