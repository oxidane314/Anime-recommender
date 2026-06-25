"""
popularity.py
--------------
Popularity-based recommendation engine.

Uses a Bayesian "weighted rating" (the IMDB formula) so that anime with
very few ratings don't outrank well-established, broadly-loved titles:

    WR = (v / (v + m)) * R + (m / (v + m)) * C

    R = average rating for the anime
    v = number of votes/members for the anime
    m = minimum votes required to be listed (e.g. 75th percentile of members)
    C = mean rating across the whole dataset
"""
from __future__ import annotations

import pandas as pd


def compute_popularity_scores(anime: pd.DataFrame, m_percentile: float = 0.75) -> pd.DataFrame:
    """Attach a `popularity_score` (weighted rating) column to the anime df."""
    df = anime.copy()
    C = df["rating"].mean()
    m = df["members"].quantile(m_percentile)

    def weighted_rating(row):
        v = row["members"]
        R = row["rating"]
        return (v / (v + m)) * R + (m / (v + m)) * C

    df["popularity_score"] = df.apply(weighted_rating, axis=1)
    return df


def recommend_popular(anime: pd.DataFrame, n: int = 10, genre_filter: str | None = None,
                       type_filter: str | None = None) -> pd.DataFrame:
    """
    Return the top-n anime by popularity score.

    Parameters
    ----------
    anime : cleaned anime dataframe (must contain rating, members columns)
    n : number of recommendations
    genre_filter : optional substring to filter genre column (e.g. "Action")
    type_filter : optional exact match on `type` column (e.g. "TV", "Movie")
    """
    scored = compute_popularity_scores(anime)

    if genre_filter:
        scored = scored[scored["genre"].str.contains(genre_filter, case=False, na=False)]
    if type_filter:
        scored = scored[scored["type"] == type_filter]

    top = scored.sort_values("popularity_score", ascending=False).head(n)
    return top[["anime_id", "name", "genre", "type", "rating", "members", "popularity_score"]].reset_index(drop=True)


if __name__ == "__main__":
    from preprocessing import build_clean_dataset

    data = build_clean_dataset()
    print(recommend_popular(data.anime, n=10))
