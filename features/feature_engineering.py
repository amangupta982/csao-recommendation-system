# features/feature_engineering.py

import pandas as pd
import numpy as np


def create_features(cart_data, users, restaurants):

    # Merge user + restaurant metadata
    df = cart_data.merge(users, on="user_id")
    df = df.merge(restaurants, on="rest_id")

    # One-hot encoding for category & cuisine
    df = pd.get_dummies(df, columns=["candidate_category", "cuisine"])

    # -----------------------------
    # Basic Category Flags
    # -----------------------------
    df["is_beverage"] = df.get("candidate_category_Beverage", 0)
    df["is_dessert"] = df.get("candidate_category_Dessert", 0)
    df["is_main"] = df.get("candidate_category_Main", 0)

    # -----------------------------
    # Time Context Features
    # -----------------------------
    df["is_late_night"] = (df["hour"] >= 22).astype(int)
    df["is_lunch_time"] = ((df["hour"] >= 12) & (df["hour"] <= 15)).astype(int)

    # -----------------------------
    # Sequential Cart Features
    # -----------------------------
    df["cart_length"] = df["cart_sequence"].apply(len)
    df["cart_unique_items"] = df["cart_sequence"].apply(lambda x: len(set(x)))

    df["candidate_in_cart"] = df.apply(
        lambda row: 1 if row["candidate_item"] in row["cart_sequence"] else 0,
        axis=1
    )

    # -----------------------------
    # Price-Based Features
    # -----------------------------
    df["price_affinity"] = abs(df["avg_order_value"] - df["candidate_price"])
    df["user_price_ratio"] = df["candidate_price"] / (df["avg_order_value"] + 1)

    # -----------------------------
    # Real Restaurant Rating Features (NEW)
    # -----------------------------
    df["rating"] = df["rating"].fillna(0)

    df["high_rating"] = (df["rating"] >= 4.0).astype(int)

    df["rating_price_interaction"] = df["rating"] * df["candidate_price"]

    # -----------------------------
    # Context Interaction Features
    # -----------------------------
    df["veg_price_interaction"] = df["veg_pref_score"] * df["candidate_price"]

    df["late_night_beverage"] = df["is_late_night"] * df["is_beverage"]

    df["lunch_main_interaction"] = df["is_lunch_time"] * df["is_main"]

    return df