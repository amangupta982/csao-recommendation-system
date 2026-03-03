# models/train_model.py

from sklearn.metrics import roc_auc_score, ndcg_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


def train_model(df):

    # Split by session
    unique_sessions = df["session_id"].unique()
    np.random.shuffle(unique_sessions)

    split_index = int(len(unique_sessions) * 0.8)

    train_sessions = unique_sessions[:split_index]
    test_sessions = unique_sessions[split_index:]

    train_df = df[df["session_id"].isin(train_sessions)]
    test_df = df[df["session_id"].isin(test_sessions)]

    X_train = train_df.drop(columns=["accepted"])
    y_train = train_df["accepted"]

    X_test = test_df.drop(columns=["accepted"])
    y_test = test_df["accepted"]

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, preds)
    ndcg = ndcg_score([y_test], [preds])

    test_df = test_df.copy()
    test_df["pred"] = preds

    precision_at_1 = []
    precision_at_5 = []

    for session in test_df["session_id"].unique():

        session_df = test_df[test_df["session_id"] == session]
        session_df = session_df.sort_values("pred", ascending=False)

        top_1 = session_df.head(1)
        top_5 = session_df.head(5)

        precision_at_1.append(top_1["accepted"].mean())
        precision_at_5.append(top_5["accepted"].mean())

    print(f"AUC Score: {auc:.4f}")
    print(f"Precision@1: {np.mean(precision_at_1):.4f}")
    print(f"Precision@5: {np.mean(precision_at_5):.4f}")
    print(f"NDCG: {ndcg:.4f}")

    # Feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    return model, X_train.columns