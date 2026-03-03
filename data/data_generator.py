# data/data_generator.py

import numpy as np
import pandas as pd
from config import *

def generate_data():
    np.random.seed(RANDOM_STATE)

    # -----------------------------
    # USERS (Synthetic - Keep Same)
    # -----------------------------
    users = pd.DataFrame({
        "user_id": range(NUM_USERS),
        "avg_order_value": np.random.normal(300, 80, NUM_USERS),
        "order_frequency": np.random.randint(1, 20, NUM_USERS),
        "veg_pref_score": np.random.uniform(0, 1, NUM_USERS)
    })

    # -----------------------------
    # RESTAURANTS (Real Dataset)
    # -----------------------------
    zomato = pd.read_csv("data/zomato.csv")

    # Basic cleaning
    zomato = zomato.dropna(subset=["rate", "approx_cost(for two people)", "cuisines"])

    # Convert rating to numeric (remove '/5')
    zomato["rate"] = zomato["rate"].astype(str).str.replace("/5", "", regex=False)
    zomato["rate"] = pd.to_numeric(zomato["rate"], errors="coerce")

    # Clean cost column (remove commas)
    zomato["approx_cost(for two people)"] = (
        zomato["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    zomato["approx_cost(for two people)"] = pd.to_numeric(
        zomato["approx_cost(for two people)"], errors="coerce"
    )

    zomato = zomato.dropna(subset=["rate", "approx_cost(for two people)"])

    # Limit size for fast hackathon runtime
    zomato = zomato.head(2000).reset_index(drop=True)

    zomato["rest_id"] = range(len(zomato))

    restaurants = pd.DataFrame({
        "rest_id": zomato["rest_id"],
        "cuisine": zomato["cuisines"],
        "price_range": zomato["approx_cost(for two people)"],
        "rating": zomato["rate"]
    })

    # -----------------------------
    # ITEMS (Synthetic - Linked to Real Restaurants)
    # -----------------------------
    categories = ["Main", "Dessert", "Beverage", "Side"]

    items = pd.DataFrame({
        "item_id": range(NUM_ITEMS),
        "rest_id": np.random.choice(restaurants["rest_id"], NUM_ITEMS),
        "category": np.random.choice(categories, NUM_ITEMS),
        "price": np.random.randint(50, 400, NUM_ITEMS)
    })

    # -----------------------------
    # SESSION + CART SIMULATION
    # -----------------------------
    data = []
    session_id = 0

    for _ in range(NUM_SESSIONS):

        session_id += 1

        user = np.random.choice(users["user_id"])
        rest = np.random.choice(restaurants["rest_id"])
        hour = np.random.randint(0, 24)

        # Simulate cart sequence
        cart_sequence = list(items.sample(3)["item_id"])

        candidate_pool = items.sample(5).reset_index(drop=True)

        scores = []

        for _, candidate in candidate_pool.iterrows():

            score = 0

            # Time-based logic
            if hour >= 22 and candidate["category"] == "Beverage":
                score += 3

            if 12 <= hour <= 15 and candidate["category"] == "Main":
                score += 2

            # Veg preference logic
            if users.loc[users["user_id"] == user, "veg_pref_score"].values[0] > 0.6:
                if candidate["category"] == "Dessert":
                    score += 2

            # Price logic
            if candidate["price"] < 200:
                score += 1

            # Avoid recommending item already in cart
            if candidate["item_id"] in cart_sequence:
                score -= 3

            scores.append(score)

        best_index = np.argmax(scores)

        for i, candidate in candidate_pool.iterrows():

            accepted = 1 if i == best_index else 0

            data.append([
                session_id,
                user,
                rest,
                hour,
                cart_sequence,
                candidate["item_id"],
                candidate["category"],
                candidate["price"],
                accepted
            ])

    cart_data = pd.DataFrame(data, columns=[
        "session_id",
        "user_id",
        "rest_id",
        "hour",
        "cart_sequence",
        "candidate_item",
        "candidate_category",
        "candidate_price",
        "accepted"
    ])

    return users, restaurants, items, cart_data