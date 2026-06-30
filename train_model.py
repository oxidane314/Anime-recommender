from src.preprocessing import build_clean_dataset
from src.collaborative import CollaborativeRecommender

print("Loading dataset...")
data = build_clean_dataset()

print("Training SVD model...")
cf = CollaborativeRecommender(data.ratings_explicit)

cf.train_svd()

print("Saving model...")
cf.save("svd_model.joblib")

print("Model saved successfully!")