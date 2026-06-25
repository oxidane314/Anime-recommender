"""
content_based.py
-----------------
Content-based filtering using TF-IDF over genre tokens (combined with
type and a binned episode count) and cosine similarity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentRecommender:
    """Builds a TF-IDF + cosine-similarity content-based recommender over anime metadata."""

    def __init__(self, anime: pd.DataFrame):
        self.anime = anime.reset_index(drop=True).copy()
        self._build_feature_soup()
        self._fit()

    def _bin_episodes(self, eps: float) -> str:
        if eps <= 1:
            return "movie_length"
        elif eps <= 13:
            return "short_series"
        elif eps <= 26:
            return "standard_series"
        elif eps <= 60:
            return "long_series"
        return "very_long_series"

    def _build_feature_soup(self):
        """Combine genre + type + episode-bucket into a single text 'soup' for TF-IDF."""
        df = self.anime
        genre_clean = df["genre"].fillna("").str.replace(",", " ").str.lower()
        type_clean = df["type"].fillna("").str.lower()
        eps_bucket = df["episodes"].apply(self._bin_episodes)

        # repeat genre tokens 2x so they dominate the TF-IDF signal over type/episode bucket
        self.anime["soup"] = (genre_clean + " " + genre_clean + " " + type_clean + " " + eps_bucket)

    def _fit(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z\-]+")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.anime["soup"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # quick lookup: lowercase name -> row index
        self._index_by_name = {
            name.lower(): idx for idx, name in enumerate(self.anime["name"])
        }

    def _resolve_index(self, anime_name: str) -> int | None:
        key = anime_name.lower().strip()
        if key in self._index_by_name:
            return self._index_by_name[key]
        # fallback: closest substring match
        matches = [n for n in self._index_by_name if key in n]
        if matches:
            return self._index_by_name[matches[0]]
        return None

    def recommend_similar(self, anime_name: str, n: int = 10) -> pd.DataFrame:
        """
        Return the top-n anime most similar to `anime_name` based on genre/type/episode
        content similarity (cosine similarity over TF-IDF vectors).
        """
        idx = self._resolve_index(anime_name)
        if idx is None:
            raise ValueError(f"'{anime_name}' not found in the dataset.")

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        result_idx = [i for i, _ in sim_scores]
        scores = [round(float(s), 4) for _, s in sim_scores]

        out = self.anime.iloc[result_idx][["anime_id", "name", "genre", "type", "episodes", "rating"]].copy()
        out["similarity_score"] = scores
        return out.reset_index(drop=True)

    def similarity_for_pair(self, idx_a: int, idx_b: int) -> float:
        return float(self.similarity_matrix[idx_a, idx_b])

    def get_index(self, anime_name: str) -> int | None:
        return self._resolve_index(anime_name)

    def content_score_for_user_profile(self, liked_anime_ids: list, n: int = 20) -> pd.DataFrame:
        """
        Build a user 'taste profile' from a list of liked anime_ids by averaging their
        TF-IDF vectors, then rank all anime by cosine similarity to that profile.
        Used by the hybrid recommender.
        """
        idxs = self.anime.index[self.anime["anime_id"].isin(liked_anime_ids)].tolist()
        if not idxs:
            return pd.DataFrame(columns=["anime_id", "name", "content_score"])

        profile_vector = np.asarray(self.tfidf_matrix[idxs].mean(axis=0))
        sims = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()

        out = self.anime[["anime_id", "name"]].copy()
        out["content_score"] = sims
        out = out[~out["anime_id"].isin(liked_anime_ids)]

        scaler = MinMaxScaler()
        out["content_score"] = scaler.fit_transform(out[["content_score"]])
        return out.sort_values("content_score", ascending=False).head(n).reset_index(drop=True)

    def explain_similarity(self, anime_a: str, anime_b: str) -> str:
        """Human-readable explanation of why two anime are similar (shared genres)."""
        idx_a = self._resolve_index(anime_a)
        idx_b = self._resolve_index(anime_b)
        if idx_a is None or idx_b is None:
            return "Insufficient metadata to explain similarity."

        genres_a = set(g.strip() for g in self.anime.iloc[idx_a]["genre"].split(","))
        genres_b = set(g.strip() for g in self.anime.iloc[idx_b]["genre"].split(","))
        shared = genres_a & genres_b

        name_b = self.anime.iloc[idx_b]["name"]
        name_a = self.anime.iloc[idx_a]["name"]
        if shared:
            return f"{name_b} is recommended because you liked {name_a}. Both share {', '.join(sorted(shared))} themes."
        return f"{name_b} is recommended because you liked {name_a}, based on overall genre/style similarity."


if __name__ == "__main__":
    from preprocessing import build_clean_dataset

    data = build_clean_dataset()
    rec = ContentRecommender(data.anime)
    print(rec.recommend_similar("Attack on Titan", n=10))
