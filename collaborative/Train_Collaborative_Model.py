import pandas as pd
from surprise import Dataset, Reader, SVD
import pickle

ratings_df = pd.read_csv("data/ratings_netflix_mapped.csv")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'title', 'rating']], reader)

trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

with open("collaborative_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ collaborative_model.pkl saved!")
