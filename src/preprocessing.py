"""
preprocessing.py
-----------------
Loading, cleaning, and basic feature preparation for the
Anime Recommendations Database (anime.csv, rating.csv).

Schema (Kaggle - CooperUnion/anime-recommendations-database):
  anime.csv:  anime_id, name, genre, type, episodes, rating, members
  rating.csv: user_id, anime_id, rating  (rating == -1 means "watched, not rated")
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ANIME_PATH = os.path.join(DATA_DIR, "anime.csv")
RATING_PATH = os.path.join(DATA_DIR, "rating.csv")


@dataclass
class CleanedData:
    anime: pd.DataFrame
    ratings: pd.DataFrame
    ratings_explicit: pd.DataFrame  # excludes the -1 "watched but unrated" rows


def load_raw(anime_path: str = ANIME_PATH, rating_path: str = RATING_PATH):
    """Load the two raw CSVs."""
    logger.info("Loading raw CSV files...")
    anime = pd.read_csv(anime_path)
    ratings = pd.read_csv(rating_path)
    logger.info(f"anime.csv shape: {anime.shape} | rating.csv shape: {ratings.shape}")
    return anime, ratings


def clean_anime(anime: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values & duplicates in the anime metadata table."""
    df = anime.copy()

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate anime rows.")

    df = df.dropna(subset=["anime_id", "name"])

    df["genre"] = df["genre"].fillna("Unknown")
    df["type"] = df["type"].fillna("Unknown")

    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    median_eps = df["episodes"].median()
    df["episodes"] = df["episodes"].fillna(median_eps)

    df = df.dropna(subset=["rating"])

    df["members"] = pd.to_numeric(df["members"], errors="coerce").fillna(0)
    df = df[df["members"] >= 0]

    df = df.reset_index(drop=True)
    logger.info(f"Cleaned anime shape: {df.shape}")
    return df


def clean_ratings(ratings: pd.DataFrame, valid_anime_ids: set) -> pd.DataFrame:
    """Clean the user-item rating interactions."""
    df = ratings.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "anime_id"])
    logger.info(f"Removed {before - len(df)} duplicate rating rows.")

    df = df.dropna(subset=["user_id", "anime_id", "rating"])
    df = df[df["anime_id"].isin(valid_anime_ids)]

    df = df.reset_index(drop=True)
    logger.info(f"Cleaned ratings shape: {df.shape}")
    return df


def build_clean_dataset(anime_path: str = ANIME_PATH, rating_path: str = RATING_PATH) -> CleanedData:
    """Full pipeline: load raw -> clean -> return CleanedData bundle."""
    anime_raw, ratings_raw = load_raw(anime_path, rating_path)
    anime = clean_anime(anime_raw)
    ratings = clean_ratings(ratings_raw, set(anime["anime_id"]))

    # -1 in the Kaggle dataset means "user watched it but didn't rate it"
    ratings_explicit = ratings[ratings["rating"] != -1].reset_index(drop=True)

    return CleanedData(anime=anime, ratings=ratings, ratings_explicit=ratings_explicit)


def basic_eda_stats(data: CleanedData) -> dict:
    """Return a dictionary of high level EDA stats (used by app + reports)."""
    anime, ratings, ratings_explicit = data.anime, data.ratings, data.ratings_explicit

    stats = {
        "n_anime": anime["anime_id"].nunique(),
        "n_users": ratings["user_id"].nunique(),
        "n_ratings_total": len(ratings),
        "n_ratings_explicit": len(ratings_explicit),
        "pct_unrated_watch": round(100 * (1 - len(ratings_explicit) / max(len(ratings), 1)), 2),
        "avg_rating_explicit": round(ratings_explicit["rating"].mean(), 2) if len(ratings_explicit) else None,
        "median_episodes": float(anime["episodes"].median()),
        "n_unique_genres": len(
            set(g.strip() for sub in anime["genre"].dropna().str.split(",") for g in sub)
        ) if "genre" in anime else 0,
        "type_distribution": anime["type"].value_counts().to_dict(),
    }
    return stats


if __name__ == "__main__":
    data = build_clean_dataset()
    print(basic_eda_stats(data))
