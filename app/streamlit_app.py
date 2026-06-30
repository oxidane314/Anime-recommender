"""
streamlit_app.py
-----------------
Production Streamlit UI for the Anime Recommendation System.

Pages:
  - Home
  - Anime Search
  - Similar Anime (content-based)
  - Personalized Recommendations (collaborative + hybrid)
  - Top Anime (popularity-based)
  - Analytics Dashboard
"""
import os
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import build_clean_dataset, basic_eda_stats
from popularity import recommend_popular
from content_based import ContentRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resource loading — runs once per server lifecycle
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading data & training models (first run only)...")
def load_system():
    data = build_clean_dataset()

    content_rec = ContentRecommender(data.anime)

    collab_rec = CollaborativeRecommender(data.ratings_explicit)

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "svd_model.joblib"
    )

    if not os.path.exists(model_path):
        st.error(
            "Trained model not found!\n\n"
            "Run train_model.py locally first."
        )
        st.stop()

    collab_rec.svd_model = CollaborativeRecommender.load("svd_model.joblib")

    hybrid_rec = HybridRecommender(
        content_rec,
        collab_rec,
        data.anime,
        w_collab=0.6,
        w_content=0.4,
    )

    return data, content_rec, collab_rec, hybrid_rec

data, content_rec, collab_rec, hybrid_rec = load_system()
anime_df = data.anime

# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🎌 Anime Recommender")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Anime Search", "Similar Anime", "Personalized Recommendations", "Top Anime", "Analytics Dashboard"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Pandas, Scikit-learn, Surprise & Streamlit")

# --------------------------------------------------------------------------- #
# HOME
# --------------------------------------------------------------------------- #
if page == "Home":
    st.title("🎌 Anime Recommendation System")
    st.markdown(
        """
        A production-style, end-to-end recommendation engine built on the
        **Anime Recommendations Database** (Kaggle), combining four
        recommendation strategies:

        | Strategy | Technique |
        |---|---|
        | **Popularity-Based** | Bayesian weighted rating (IMDB formula) |
        | **Content-Based Filtering** | TF-IDF on genres + cosine similarity |
        | **Collaborative Filtering** | KNNBasic & SVD (matrix factorization) via `surprise` |
        | **Hybrid** | Weighted blend: `0.6 * collaborative + 0.4 * content` |

        Every recommendation comes with a **human-readable explanation** of *why*
        it was suggested.
        """
    )

    stats = basic_eda_stats(data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anime Titles", stats["n_anime"])
    c2.metric("Users", stats["n_users"])
    c3.metric("Ratings Collected", stats["n_ratings_total"])
    c4.metric("Avg. Rating", stats["avg_rating_explicit"])

    st.markdown("---")
    st.subheader("How to use this app")
    st.markdown(
        """
        - **Anime Search** — browse and filter the catalog.
        - **Similar Anime** — pick a title you love, get content-based lookalikes.
        - **Personalized Recommendations** — enter a user ID for collaborative & hybrid picks.
        - **Top Anime** — popularity leaderboard, filterable by genre/type.
        - **Analytics Dashboard** — full EDA visualizations.
        """
    )

# --------------------------------------------------------------------------- #
# ANIME SEARCH
# --------------------------------------------------------------------------- #
elif page == "Anime Search":
    st.title("🔍 Anime Search")
    query = st.text_input("Search by title")
    col1, col2 = st.columns(2)
    genre_filter = col1.selectbox(
        "Filter by genre",
        ["All"] + sorted(set(g.strip() for sub in anime_df["genre"].dropna().str.split(",") for g in sub)),
    )
    type_filter = col2.selectbox("Filter by type", ["All"] + sorted(anime_df["type"].dropna().unique().tolist()))

    results = anime_df.copy()
    if query:
        results = results[results["name"].str.contains(query, case=False, na=False)]
    if genre_filter != "All":
        results = results[results["genre"].str.contains(genre_filter, case=False, na=False)]
    if type_filter != "All":
        results = results[results["type"] == type_filter]

    st.write(f"**{len(results)}** anime found")
    st.dataframe(
        results[["name", "genre", "type", "episodes", "rating", "members"]].sort_values("rating", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------------------------------- #
# SIMILAR ANIME (Content-Based)
# --------------------------------------------------------------------------- #
elif page == "Similar Anime":
    st.title("🧩 Similar Anime (Content-Based)")
    st.caption("Recommendations powered by TF-IDF over genre/type/episode-length, ranked by cosine similarity.")

    anime_name = st.selectbox("Pick an anime you like", sorted(anime_df["name"].unique()))
    n = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Get Similar Anime", type="primary"):
        try:
            recs = content_rec.recommend_similar(anime_name, n=n)
            for _, row in recs.iterrows():
                with st.container(border=True):
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"**{row['name']}**  \n{row['genre']}  \n*{row['type']} · {int(row['episodes'])} eps · ⭐ {row['rating']}*")
                    cols[1].metric("Similarity", f"{row['similarity_score']:.2f}")
                    st.caption(content_rec.explain_similarity(anime_name, row["name"]))
        except ValueError as e:
            st.error(str(e))

# --------------------------------------------------------------------------- #
# PERSONALIZED RECOMMENDATIONS (Collaborative + Hybrid)
# --------------------------------------------------------------------------- #
elif page == "Personalized Recommendations":
    st.title("🎯 Personalized Recommendations")

    valid_users = sorted(data.ratings_explicit["user_id"].unique())
    user_id = st.selectbox("Select a user ID", valid_users)
    n = st.slider("Number of recommendations", 5, 20, 10)
    mode = st.radio("Recommendation mode", ["Hybrid (recommended)", "Collaborative Filtering only"], horizontal=True)

    if st.button("Generate Recommendations", type="primary"):
        if mode.startswith("Hybrid"):
            recs = hybrid_rec.recommend_for_user(user_id, n=n)
            for _, row in recs.iterrows():
                with st.container(border=True):
                    cols = st.columns([3, 1, 1, 1])
                    cols[0].markdown(f"**{row['name']}**  \n{row['genre']} · {row['type']}")
                    cols[1].metric("Collab", row["collab_score"])
                    cols[2].metric("Content", row["content_score"])
                    cols[3].metric("Final", row["final_score"])
                    st.caption(f"💡 {row['explanation']}")
        else:
            recs = collab_rec.recommend_for_user(user_id, anime_df, n=n)
            st.dataframe(recs, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# TOP ANIME (Popularity-Based)
# --------------------------------------------------------------------------- #
elif page == "Top Anime":
    st.title("🏆 Top Anime (Popularity-Based)")
    st.caption("Ranked by Bayesian weighted rating, balancing average score against number of voters.")

    col1, col2, col3 = st.columns(3)
    n = col1.slider("How many?", 5, 30, 10)
    genre_filter = col2.selectbox(
        "Genre filter",
        ["All"] + sorted(set(g.strip() for sub in anime_df["genre"].dropna().str.split(",") for g in sub)),
        key="top_genre",
    )
    type_filter = col3.selectbox("Type filter", ["All"] + sorted(anime_df["type"].dropna().unique().tolist()), key="top_type")

    recs = recommend_popular(
        anime_df, n=n,
        genre_filter=None if genre_filter == "All" else genre_filter,
        type_filter=None if type_filter == "All" else type_filter,
    )
    st.dataframe(recs, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# ANALYTICS DASHBOARD
# --------------------------------------------------------------------------- #
elif page == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    sns.set_theme(style="whitegrid")

    stats = basic_eda_stats(data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Genres", stats["n_unique_genres"])
    c2.metric("Median Episodes", stats["median_episodes"])
    c3.metric("% Watched-Not-Rated", f"{stats['pct_unrated_watch']}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Ratings", "Genres", "Popularity", "User Activity"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(anime_df["rating"], bins=30, kde=True, ax=ax, color="steelblue")
        ax.set_title("Distribution of Anime Average Ratings")
        st.pyplot(fig)

    with tab2:
        genre_series = anime_df["genre"].dropna().str.split(", ").explode()
        genre_counts = genre_series.value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, palette="viridis")
        ax.set_title("Top 15 Genres")
        st.pyplot(fig)

    with tab3:
        top_pop = anime_df.sort_values("members", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="members", y="name", data=top_pop, ax=ax, palette="mako")
        ax.set_title("Top 10 Anime by Member Count")
        st.pyplot(fig)

    with tab4:
        user_activity = data.ratings.groupby("user_id").size()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(user_activity, bins=30, color="purple", ax=ax)
        ax.set_title("Ratings per User Distribution")
        st.pyplot(fig)
        st.write(f"Median ratings/user: **{user_activity.median():.0f}** | Max: **{user_activity.max()}**")
