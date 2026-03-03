# utils/recommend.py

import numpy as np

def generate_candidates(df, rest_id, top_n=20):
    """
    Stage 1: Candidate Generation
    Select top items from same restaurant.
    """
    candidates = df[df["rest_id"] == rest_id].copy()
    return candidates.head(top_n)


def recommend_top_k(model, df, feature_columns, user_id, rest_id, k=5):

    # Stage 1
    candidates = generate_candidates(df, rest_id)

    if len(candidates) == 0:
        print("No restaurant data found.")
        return None

    # Stage 2: Ranking
    candidates["user_id"] = user_id

    X_candidates = candidates[feature_columns]
    X_candidates = X_candidates.select_dtypes(include=[np.number])

    candidates["score"] = model.predict_proba(X_candidates)[:, 1]

    ranked = candidates.sort_values("score", ascending=False)

    return ranked.head(k)