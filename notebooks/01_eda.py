# %% [markdown]
# # Anime Recommendation System — Exploratory Data Analysis
#
# This notebook explores the **Anime Recommendations Database** (anime.csv, rating.csv)
# before any model building: missing values, duplicates, rating distributions,
# genre distributions, popularity, and user activity patterns.

# %%
import sys
sys.path.append("../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import build_clean_dataset, basic_eda_stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 1. Load & Clean Data

# %%
data = build_clean_dataset()
anime, ratings, ratings_explicit = data.anime, data.ratings, data.ratings_explicit
print("Anime shape:", anime.shape)
print("Ratings shape:", ratings.shape)
print("Explicit ratings shape:", ratings_explicit.shape)
anime.head()

# %% [markdown]
# ## 2. Missing Values & Duplicates
# Missing values and duplicates are already handled inside `preprocessing.build_clean_dataset()`.
# Below we confirm the cleaned tables are free of nulls in the key columns.

# %%
print("Anime null counts:\n", anime.isnull().sum())
print("\nRatings null counts:\n", ratings.isnull().sum())

# %% [markdown]
# ## 3. Summary Statistics

# %%
stats = basic_eda_stats(data)
for k, v in stats.items():
    print(f"{k}: {v}")

# %% [markdown]
# ## 4. Rating Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(anime["rating"], bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Distribution of Anime Average Ratings")
axes[0].set_xlabel("Average Rating")

sns.histplot(ratings_explicit["rating"], bins=10, kde=False, ax=axes[1], color="coral")
axes[1].set_title("Distribution of Individual User Ratings (Explicit)")
axes[1].set_xlabel("User Rating (1-10)")
plt.tight_layout()
plt.savefig("../reports/rating_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Genre Distribution

# %%
genre_series = anime["genre"].dropna().str.split(", ").explode()
genre_counts = genre_series.value_counts().head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis")
plt.title("Top 20 Most Common Anime Genres")
plt.xlabel("Number of Anime")
plt.tight_layout()
plt.savefig("../reports/genre_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Most Popular Anime (by Members)

# %%
most_popular = anime.sort_values("members", ascending=False).head(15)
plt.figure(figsize=(10, 8))
sns.barplot(x="members", y="name", data=most_popular, palette="mako")
plt.title("Top 15 Most Popular Anime (by Member Count)")
plt.xlabel("Number of Members")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/most_popular_anime.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Highest Rated Anime (min member threshold to avoid noise)

# %%
member_threshold = anime["members"].quantile(0.5)
highest_rated = anime[anime["members"] >= member_threshold].sort_values("rating", ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x="rating", y="name", data=highest_rated, palette="rocket")
plt.title("Top 15 Highest Rated Anime (min. popularity filter applied)")
plt.xlabel("Average Rating")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/highest_rated_anime.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Anime Type Distribution

# %%
plt.figure(figsize=(8, 5))
anime["type"].value_counts().plot(kind="bar", color="teal")
plt.title("Anime Count by Type")
plt.xlabel("Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../reports/type_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. User Activity Analysis

# %%
user_activity = ratings.groupby("user_id").size().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.histplot(user_activity, bins=40, color="purple")
plt.title("Distribution of Number of Ratings per User")
plt.xlabel("Number of Ratings Given")
plt.tight_layout()
plt.savefig("../reports/user_activity_distribution.png", dpi=150)
plt.show()

print("Median ratings per user:", user_activity.median())
print("Max ratings by a single user:", user_activity.max())
print("Users with only 1 rating:", (user_activity == 1).sum())

# %% [markdown]
# ## 10. Rating vs. Popularity (Members) Correlation

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x="members", y="rating", data=anime, alpha=0.6)
plt.xscale("log")
plt.title("Rating vs. Popularity (log scale members)")
plt.xlabel("Members (log scale)")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig("../reports/rating_vs_popularity.png", dpi=150)
plt.show()

print("Correlation (rating, members):", anime["rating"].corr(anime["members"]))

# %% [markdown]
# ## Key Takeaways
# - Most anime ratings cluster in the 6.5–8.5 range; the tail of very high ratings
#   belongs to long-running, high-member-count titles.
# - Action, Comedy, and Drama dominate genre frequency.
# - Popularity (members) and rating are positively but weakly correlated — meaning
#   pure popularity-based recommendations alone are not a substitute for personalization.
# - User activity is highly skewed: a small fraction of users contribute a
#   disproportionate share of ratings (classic long-tail behavior), reinforcing the
#   need for collaborative + content hybrid approaches to handle sparse users well.
