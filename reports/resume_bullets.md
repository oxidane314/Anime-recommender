# Resume Bullet Points — Anime Recommendation System

Use 3–5 of these on your resume/LinkedIn depending on space. Swap in your real
metrics from `reports/performance_report.md` once run on the full dataset.

## Project Title
**Anime Recommendation System** | Python, Scikit-learn, Surprise, Streamlit

## Bullet Points (ATS-friendly, quantified)

- Engineered an end-to-end hybrid recommendation system in Python combining
  popularity-based, content-based (TF-IDF + cosine similarity), and collaborative
  filtering (SVD matrix factorization, KNN) models across 12,000+ anime titles and
  7M+ user ratings.

- Built and benchmarked SVD and KNN collaborative-filtering models using the Surprise
  library, achieving an RMSE of ~1.25 and improving rating-prediction accuracy by ~15%
  over the item-based KNN baseline.

- Designed a weighted hybrid recommendation algorithm (60% collaborative / 40%
  content-based) that mitigates cold-start problems for new users, improving
  recommendation Precision@10 by combining taste-profile similarity with collaborative signal.

- Implemented an explainable-AI layer generating natural-language justifications for
  every recommendation, increasing transparency and user trust in model output.

- Developed a 6-page interactive Streamlit web application (search, content-based
  discovery, personalized recommendations, popularity leaderboard, analytics
  dashboard) deployed for public demonstration.

- Conducted full exploratory data analysis on rating distributions, genre frequency,
  and user activity skew, informing feature engineering decisions and bias-aware
  popularity scoring (Bayesian weighted rating).

- Established a reusable evaluation framework computing RMSE, MAE, Precision@K, and
  Recall@K across all four recommendation strategies, producing an automated Markdown
  performance report for stakeholder review.

- Structured the project using industry-standard ML repository conventions
  (modular `src/`, `notebooks/`, `app/`, `reports/`) to support maintainability and
  collaborative development on GitHub.

## Short Version (for tight resume space)

- Built a hybrid Anime Recommendation System (Python, Scikit-learn, Surprise,
  Streamlit) combining content-based and collaborative filtering (SVD/KNN) across
  12K+ titles and 7M+ ratings; achieved RMSE ~1.25 and deployed an explainable,
  6-page Streamlit web app.
