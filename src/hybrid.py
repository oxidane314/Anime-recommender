"""
hybrid.py
---------
Hybrid recommender: blends collaborative-filtering predicted ratings with
content-based similarity-to-user-taste-profile scores.

    final_score = w_collab * collaborative_score + w_content * content_score

Both component scores are min-max normalized to [0, 1] before blending so
that the weights are meaningfully comparable.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from content_based import ContentRecommender
from collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(self, content_rec: ContentRecommender, collab_rec: CollaborativeRecommender,
                 anime_df: pd.DataFrame, w_collab: float = 0.6, w_content: float = 0.4):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.anime_df = anime_df
        self.w_collab = w_collab
        self.w_content = w_content

    def _user_liked_anime(self, user_id: int, like_threshold: float = 7.0) -> list:
        """Anime a user rated >= like_threshold, used to build their content taste profile."""
        user_ratings = self.collab_rec.ratings[self.collab_rec.ratings.user_id == user_id]
        liked = user_ratings[user_ratings.rating >= like_threshold]["anime_id"].tolist()
        return liked

    def recommend_for_user(self, user_id: int, n: int = 10, like_threshold: float = 7.0) -> pd.DataFrame:
        """
        Produce blended recommendations for a user:
          1. Collaborative score = SVD-predicted rating (normalized 0-1)
          2. Content score = cosine similarity to the user's liked-anime taste profile (0-1)
          3. final_score = w_collab * collab_score + w_content * content_score
        """
        liked = self._user_liked_anime(user_id, like_threshold)

        watched = set(self.collab_rec.ratings.loc[self.collab_rec.ratings.user_id == user_id, "anime_id"])
        candidates = [aid for aid in self.anime_df["anime_id"] if aid not in watched]

        # --- collaborative component ---
        collab_preds = {aid: self.collab_rec.svd_model.predict(user_id, aid).est for aid in candidates}
        collab_df = pd.DataFrame(list(collab_preds.items()), columns=["anime_id", "collab_raw"])
        scaler = MinMaxScaler()
        collab_df["collab_score"] = scaler.fit_transform(collab_df[["collab_raw"]])

        # --- content component ---
        if liked:
            content_df = self.content_rec.content_score_for_user_profile(liked, n=len(candidates) + 1)
            content_df = content_df[["anime_id", "content_score"]]
        else:
            # cold-start: no content signal available, fall back to neutral score
            content_df = pd.DataFrame({"anime_id": candidates, "content_score": 0.5})

        merged = collab_df.merge(content_df, on="anime_id", how="left")
        merged["content_score"] = merged["content_score"].fillna(0.0)

        merged["final_score"] = (
            self.w_collab * merged["collab_score"] + self.w_content * merged["content_score"]
        )

        merged = merged.merge(self.anime_df[["anime_id", "name", "genre", "type", "rating"]], on="anime_id")
        merged = merged.sort_values("final_score", ascending=False).head(n).reset_index(drop=True)

        # attach explanation
        seed_anime_name = None
        if liked:
            seed_id = liked[0]
            seed_anime_name = self.anime_df.loc[self.anime_df.anime_id == seed_id, "name"].values
            seed_anime_name = seed_anime_name[0] if len(seed_anime_name) else None

        explanations = []
        for _, row in merged.iterrows():
            if seed_anime_name:
                explanations.append(self.content_rec.explain_similarity(seed_anime_name, row["name"]))
            else:
                explanations.append(
                    f"{row['name']} is recommended based on predicted rating patterns from similar users."
                )
        merged["explanation"] = explanations

        merged["final_score"] = merged["final_score"].round(4)
        merged["collab_score"] = merged["collab_score"].round(4)
        merged["content_score"] = merged["content_score"].round(4)

        return merged[["anime_id", "name", "genre", "type", "rating",
                        "collab_score", "content_score", "final_score", "explanation"]]


if __name__ == "__main__":
    from preprocessing import build_clean_dataset

    data = build_clean_dataset()
    content_rec = ContentRecommender(data.anime)
    collab_rec = CollaborativeRecommender(data.ratings_explicit)
    collab_rec.train_svd()

    hybrid = HybridRecommender(content_rec, collab_rec, data.anime, w_collab=0.6, w_content=0.4)
    sample_user = data.ratings_explicit["user_id"].iloc[0]
    print(hybrid.recommend_for_user(sample_user, n=10))
