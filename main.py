# main.py

import time

from data.data_generator import generate_data
from features.feature_engineering import create_features
from models.train_model import train_model
from utils.recommend import recommend_top_k


def main():

    print("Generating Data...")
    users, restaurants, items, cart_data = generate_data()

    print("Creating Features...")
    df = create_features(cart_data, users, restaurants)

    print("Training Model...")
    model, feature_columns = train_model(df)

    # -------------------------------
    # Simulate Real-Time Inference
    # -------------------------------

    print("\nSimulating Real-Time Inference...")

    start = time.time()

    # Simulate feature building latency (Redis / Feature Store)
    time.sleep(0.005)

    # Simulate candidate retrieval latency (ANN / Filtering)
    time.sleep(0.015)

    # Simulate ranking latency (Model inference)
    time.sleep(0.01)

    end = time.time()

    print(f"Simulated Inference Latency: {(end - start) * 1000:.2f} ms")

    # -------------------------------
    # Example Recommendation
    # -------------------------------

    print("\nGenerating Sample Recommendations...")

    sample_user = 10
    sample_rest = 5

    recommendations = recommend_top_k(
        model,
        df,
        feature_columns,
        user_id=sample_user,
        rest_id=sample_rest,
        k=5
    )

    if recommendations is not None:
        print(recommendations[[
            "session_id",
            "user_id",
            "rest_id",
            "candidate_item",
            "candidate_price"
        ]].head())


if __name__ == "__main__":
    main()