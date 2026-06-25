# Model Performance Report

## Collaborative Filtering Model Comparison

| model                      |   RMSE |    MAE |
|:---------------------------|-------:|-------:|
| SVD (Matrix Factorization) | 1.2531 | 1.0141 |
| KNNBasic (Item-Based)      | 1.4806 | 1.2038 |


## Best Collaborative Model — Detailed Metrics

- **RMSE**: 1.2601
- **MAE**: 1.0172
- **Precision@10**: 0.7355
- **Recall@10**: 0.9659


## Content-Based Filtering — Metrics

- **Hit-Rate@10**: 0.3
- **n_users_evaluated**: 50


## Interpretation

- RMSE/MAE measure rating-prediction accuracy for collaborative filtering (lower is better).
- Precision@K / Recall@K / Hit-Rate@K measure ranking quality of the top-K recommendation list (higher is better).
- The hybrid model is expected to outperform either individual approach on ranking quality, since it combines taste-profile similarity with collaborative signal, while also mitigating cold-start problems for new users via content scores.