"""
evaluation.py
-------------
Evaluation utilities:
  - RMSE / MAE for collaborative filtering (via surprise.accuracy, wrapped here too)
  - Precision@K / Recall@K for content-based & collaborative top-N recommendations
  - A combined performance report generator
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
from surprise import accuracy


def precision_recall_at_k(predictions, k: int = 10, threshold: float = 7.0):
    """
    Compute Precision@K and Recall@K averaged over all users, given a list of
    surprise Prediction objects (from model.test(testset)).

    A recommendation is considered "relevant" if the true rating >= threshold.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum(true_r >= threshold for (_, true_r) in ratings)
        n_rec_k = min(k, len(ratings))
        top_k = ratings[:n_rec_k]
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    avg_precision = sum(precisions.values()) / len(precisions) if precisions else 0
    avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0
    return round(avg_precision, 4), round(avg_recall, 4)


def evaluate_collaborative_model(model, testset, k: int = 10, threshold: float = 7.0) -> dict:
    preds = model.test(testset)
    rmse = round(accuracy.rmse(preds, verbose=False), 4)
    mae = round(accuracy.mae(preds, verbose=False), 4)
    precision, recall = precision_recall_at_k(preds, k=k, threshold=threshold)
    return {"RMSE": rmse, "MAE": mae, f"Precision@{k}": precision, f"Recall@{k}": recall}


def evaluate_content_based(content_rec, ratings_explicit: pd.DataFrame, anime_df: pd.DataFrame,
                            k: int = 10, like_threshold: float = 7.0, n_eval_users: int = 50) -> dict:
    """
    Leave-one-out style evaluation for content-based filtering:
    For each sampled user with >=2 liked anime, hide one liked anime, use the
    remaining liked anime to build a taste profile, then check whether the
    held-out anime appears in the top-K content recommendations.
    """
    users = ratings_explicit["user_id"].unique()
    sample_users = users[:n_eval_users]

    hits, total = 0, 0
    for uid in sample_users:
        liked = ratings_explicit[(ratings_explicit.user_id == uid) &
                                  (ratings_explicit.rating >= like_threshold)]["anime_id"].tolist()
        if len(liked) < 2:
            continue

        held_out = liked[-1]
        seed = liked[:-1]

        recs = content_rec.content_score_for_user_profile(seed, n=k)
        total += 1
        if held_out in recs["anime_id"].values:
            hits += 1

    precision_at_k = round(hits / total, 4) if total else 0.0
    return {f"Hit-Rate@{k}": precision_at_k, "n_users_evaluated": total}


def generate_performance_report(collab_metrics: dict, content_metrics: dict,
                                 model_comparison_df: pd.DataFrame) -> str:
    """Build a markdown-formatted performance report string."""
    lines = ["# Model Performance Report\n"]
    lines.append("## Collaborative Filtering Model Comparison\n")
    lines.append(model_comparison_df.to_markdown(index=False))
    lines.append("\n\n## Best Collaborative Model — Detailed Metrics\n")
    for k, v in collab_metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n\n## Content-Based Filtering — Metrics\n")
    for k, v in content_metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n\n## Interpretation\n")
    lines.append(
        "- RMSE/MAE measure rating-prediction accuracy for collaborative filtering "
        "(lower is better).\n"
        "- Precision@K / Recall@K / Hit-Rate@K measure ranking quality of the top-K "
        "recommendation list (higher is better).\n"
        "- The hybrid model is expected to outperform either individual approach on "
        "ranking quality, since it combines taste-profile similarity with collaborative "
        "signal, while also mitigating cold-start problems for new users via content scores."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    from preprocessing import build_clean_dataset
    from collaborative import CollaborativeRecommender
    from content_based import ContentRecommender

    data = build_clean_dataset()
    cf = CollaborativeRecommender(data.ratings_explicit)
    trainset, testset = cf.train_test_split_data()
    svd = cf.train_svd(trainset=trainset)
    collab_metrics = evaluate_collaborative_model(svd, testset, k=10)
    cf.train_svd()  # retrain full

    content_rec = ContentRecommender(data.anime)
    content_metrics = evaluate_content_based(content_rec, data.ratings_explicit, data.anime, k=10)

    comparison = cf.compare_models()
    report = generate_performance_report(collab_metrics, content_metrics, comparison)

    with open("../reports/performance_report.md", "w") as f:
        f.write(report)

    print(report)
