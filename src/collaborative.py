"""
collaborative.py
-----------------
Collaborative filtering using the `surprise` library:
  - KNNBasic (item-based, cosine similarity)
  - SVD (matrix factorization)

Both models are trained on the explicit (non -1) user-item rating matrix.
"""
from __future__ import annotations

import os
import joblib
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
from surprise.model_selection import train_test_split

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class CollaborativeRecommender:
    def __init__(self, ratings_explicit: pd.DataFrame, rating_scale=(1, 10)):
        self.ratings = ratings_explicit
        self.reader = Reader(rating_scale=rating_scale)
        self.dataset = Dataset.load_from_df(self.ratings[["user_id", "anime_id", "rating"]], self.reader)
        self.svd_model = None
        self.knn_model = None
        self.trainset_full = self.dataset.build_full_trainset()

    # ------------------------------------------------------------------ #
    # Training & evaluation
    # ------------------------------------------------------------------ #
    def train_test_split_data(self, test_size: float = 0.2, random_state: int = 42):
        return train_test_split(self.dataset, test_size=test_size, random_state=random_state)

    def train_svd(self, n_factors: int = 50, n_epochs: int = 20, lr_all: float = 0.005,
                  reg_all: float = 0.02, trainset=None):
        trainset = trainset or self.trainset_full
        self.svd_model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        self.svd_model.fit(trainset)
        return self.svd_model

    def train_knn(self, k: int = 20, user_based: bool = False, trainset=None):
        trainset = trainset or self.trainset_full
        sim_options = {"name": "cosine", "user_based": user_based}
        self.knn_model = KNNBasic(k=k, sim_options=sim_options, verbose=False)
        self.knn_model.fit(trainset)
        return self.knn_model

    def evaluate(self, model, testset) -> dict:
        preds = model.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        mae = accuracy.mae(preds, verbose=False)
        return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}

    def compare_models(self, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """Train both algorithms on the same split and compare RMSE/MAE."""
        trainset, testset = self.train_test_split_data(test_size, random_state)

        svd = self.train_svd(trainset=trainset)
        svd_metrics = self.evaluate(svd, testset)

        knn = self.train_knn(trainset=trainset)
        knn_metrics = self.evaluate(knn, testset)

        # retrain on full data for production use after comparison
        self.train_svd()
        self.train_knn()

        return pd.DataFrame([
            {"model": "SVD (Matrix Factorization)", **svd_metrics},
            {"model": "KNNBasic (Item-Based)", **knn_metrics},
        ])

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_rating(self, user_id: int, anime_id: int, model=None) -> float:
        model = model or self.svd_model
        if model is None:
            raise RuntimeError("Model not trained yet. Call train_svd() or train_knn() first.")
        return model.predict(user_id, anime_id).est

    def recommend_for_user(self, user_id: int, anime_df: pd.DataFrame, n: int = 10,
                            model=None, exclude_watched: bool = True) -> pd.DataFrame:
        """
        Generate top-n personalized recommendations for a given user_id by predicting
        ratings on all anime they haven't rated yet, then ranking by predicted rating.
        """
        model = model or self.svd_model
        if model is None:
            raise RuntimeError("Model not trained yet. Call train_svd() first.")

        all_anime_ids = set(anime_df["anime_id"])
        if exclude_watched:
            watched = set(self.ratings.loc[self.ratings.user_id == user_id, "anime_id"])
            candidates = all_anime_ids - watched
        else:
            candidates = all_anime_ids

        preds = [(aid, model.predict(user_id, aid).est) for aid in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:n]

        ids = [p[0] for p in top]
        scores = {p[0]: round(p[1], 3) for p in top}

        out = anime_df[anime_df["anime_id"].isin(ids)][["anime_id", "name", "genre", "type", "rating"]].copy()
        out["predicted_rating"] = out["anime_id"].map(scores)
        out = out.sort_values("predicted_rating", ascending=False).reset_index(drop=True)
        return out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, name: str = "svd_model.joblib"):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, name)
        joblib.dump(self.svd_model, path)
        return path

    @staticmethod
    def load(name: str = "svd_model.joblib"):
        path = os.path.join(MODELS_DIR, name)
        return joblib.load(path)


if __name__ == "__main__":
    from preprocessing import build_clean_dataset

    data = build_clean_dataset()
    cf = CollaborativeRecommender(data.ratings_explicit)
    comparison = cf.compare_models()
    print(comparison)

    sample_user = data.ratings_explicit["user_id"].iloc[0]
    print(f"\nRecommendations for user {sample_user}:")
    print(cf.recommend_for_user(sample_user, data.anime, n=10))
