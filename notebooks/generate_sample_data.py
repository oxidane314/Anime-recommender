"""
generate_sample_data.py
------------------------
Generates a small synthetic sample mimicking the Kaggle
"Anime Recommendations Database" schema (anime.csv, rating.csv)
so the full pipeline can be run/tested before plugging in the
real dataset.

Real dataset: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
Download anime.csv and rating.csv and place them in data/ to replace
these sample files for production use.
"""
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

GENRES_POOL = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Sci-Fi",
    "Romance", "Slice of Life", "Horror", "Mystery", "Psychological",
    "Military", "Historical", "Supernatural", "Sports", "Mecha",
    "Music", "School", "Shounen", "Seinen", "Thriller"
]

TYPES = ["TV", "Movie", "OVA", "Special", "ONA"]

ANIME_NAMES = [
    "Attack on Titan", "Vinland Saga", "Death Note", "Fullmetal Alchemist: Brotherhood",
    "Naruto", "Naruto Shippuden", "One Piece", "Demon Slayer", "Jujutsu Kaisen",
    "My Hero Academia", "Hunter x Hunter", "Code Geass", "Steins;Gate",
    "Cowboy Bebop", "Tokyo Ghoul", "Mob Psycho 100", "Re:Zero",
    "Violet Evergarden", "Your Lie in April", "Clannad", "Haikyuu!!",
    "Kuroko's Basketball", "Sword Art Online", "Fairy Tail", "Bleach",
    "Dragon Ball Z", "One Punch Man", "Parasyte", "Erased", "The Promised Neverland",
    "Black Clover", "Dr. Stone", "Spy x Family", "Chainsaw Man", "Made in Abyss",
    "Psycho-Pass", "Monster", "Berserk", "Gintama", "Slam Dunk",
    "K-On!", "Toradora!", "Nisekoi", "Horimiya", "Kaguya-sama: Love is War",
    "A Silent Voice (Movie)", "Your Name (Movie)", "Spirited Away (Movie)",
    "Princess Mononoke (Movie)", "Grave of the Fireflies (Movie)",
    "Akira (Movie)", "Ghost in the Shell (Movie)", "Perfect Blue (Movie)",
    "Neon Genesis Evangelion", "Evangelion 3.0+1.0 (Movie)", "FLCL",
    "Soul Eater", "Noragami", "Black Butler", "Durarara!!",
]

rows = []
for i, name in enumerate(ANIME_NAMES, start=1):
    n_genres = random.randint(2, 4)
    genres = ", ".join(sorted(random.sample(GENRES_POOL, n_genres)))
    anime_type = random.choice(TYPES)
    episodes = random.choice([1, 12, 13, 24, 25, 26, 50, 64, 100, 148, 220])
    if anime_type == "Movie":
        episodes = 1
    rating = round(np.random.normal(7.5, 0.9), 2)
    rating = float(np.clip(rating, 5.0, 9.7))
    members = int(np.random.lognormal(mean=10, sigma=1.3))
    rows.append([i, name, genres, anime_type, episodes, rating, members])

anime_df = pd.DataFrame(rows, columns=["anime_id", "name", "genre", "type", "episodes", "rating", "members"])
anime_df.to_csv("/home/claude/Anime-Recommender/data/anime.csv", index=False)

# Synthetic user-item ratings (user_id, anime_id, rating[-1,1..10])
n_users = 500
records = []
for user_id in range(1, n_users + 1):
    n_rated = random.randint(10, 40)
    rated_anime = random.sample(range(1, len(ANIME_NAMES) + 1), min(n_rated, len(ANIME_NAMES)))
    # give each user a few "taste clusters" by biasing toward some genres
    for anime_id in rated_anime:
        if random.random() < 0.08:
            r = -1  # watched but not rated
        else:
            base = anime_df.loc[anime_df.anime_id == anime_id, "rating"].values[0]
            r = int(np.clip(round(np.random.normal(base, 1.2)), 1, 10))
        records.append([user_id, anime_id, r])

rating_df = pd.DataFrame(records, columns=["user_id", "anime_id", "rating"])
rating_df.to_csv("/home/claude/Anime-Recommender/data/rating.csv", index=False)

print("anime.csv:", anime_df.shape)
print("rating.csv:", rating_df.shape)
