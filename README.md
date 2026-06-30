# 🎌 Anime Recommendation System

A machine learning project that recommends anime using multiple recommendation techniques instead of relying on a single algorithm. The project combines popularity-based recommendations, content similarity, collaborative filtering, and a hybrid model, all wrapped inside an interactive Streamlit web application.

The goal was to understand how modern recommendation systems work end-to-end—from cleaning real-world data to training models, evaluating them, and deploying an application that users can interact with.

---

## 🚀 Live Demo

**Streamlit App:** https://your-streamlit-link.streamlit.app/

---

## Dashboard Preview

### Home Page

![Home](assets/home_page.png)

### Similar Anime Recommendations

![Similar Anime](assets/similar_anime.png)

### Personalized Recommendations

![Personalized](assets/personalized.png)

### Analytics Dashboard

![Analytics](assets/analytics_dashboard.png)

---

## Project Overview

This project uses the Anime Recommendations Database from Kaggle, containing over **12,000 anime titles** and **7 million user ratings**.

Instead of building only one recommendation algorithm, I implemented and compared multiple approaches before combining them into a hybrid recommender.

The application allows users to:

* Search for anime
* Find similar titles
* Get personalized recommendations
* Browse top-rated anime
* Explore dataset analytics through interactive visualizations

---

## Recommendation Models

### ⭐ Popularity-Based

Ranks anime using the IMDb weighted rating formula to avoid bias toward titles with very few ratings.

---

### 🎭 Content-Based Filtering

Uses:

* TF-IDF Vectorization
* Cosine Similarity

Anime are represented using genres, type, and episode information. Users receive recommendations based on similarity between anime rather than community ratings.

---

### 👥 Collaborative Filtering

Built using the Surprise library.

Implemented:

* KNNBasic
* SVD Matrix Factorization

The collaborative model predicts ratings for unseen anime based on historical user preferences.

---

### 🔀 Hybrid Recommendation System

Combines both collaborative filtering and content similarity:

Final Score =

0.6 × Collaborative Score + 0.4 × Content Score

For new users with little rating history, the system automatically falls back to the content-based recommender.

---

## Data Processing

The raw dataset required several preprocessing steps before training.

Some of the work included:

* Handling missing values
* Cleaning inconsistent episode information
* Removing duplicate records
* Converting data types
* Separating explicit ratings from implicit interactions
* Building user profiles for personalized recommendations

---

## Model Evaluation

The recommendation models were evaluated using:

* RMSE
* MAE
* Precision@K
* Recall@K
* Hit Rate@K

Among the collaborative models, **SVD consistently achieved lower prediction error than KNNBasic**, making it the default collaborative model used in the application.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Surprise
* Streamlit
* Matplotlib
* Seaborn
* Git & GitHub

---

## Project Structure

```text
Anime-Recommender/
│
├── app/
├── src/
├── data/
├── notebooks/
├── models/
├── reports/
├── assets/
├── requirements.txt
└── README.md
```

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/yourusername/Anime-Recommender.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
python -m streamlit run app/streamlit_app.py
```

---

## What I Learned

Building this project helped me understand:

* Different recommendation system approaches and when each is useful
* Feature engineering for recommendation tasks
* Model evaluation beyond simple accuracy metrics
* Building an end-to-end ML application with Streamlit
* Deploying machine learning projects for real users

---

## Future Improvements

* Sentence Transformer embeddings
* Implicit feedback models
* Docker deployment
* User authentication
* Better hybrid ranking strategy

---

## Dataset

Anime Recommendations Database

https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

---

If you found this project interesting, feel free to ⭐ the repository.
