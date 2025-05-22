import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("netflix_clean.csv")
with open("bert_embeddings.pkl", "rb") as f:
    bert_embeddings = pickle.load(f)
with open("collaborative_model.pkl", "rb") as f:
    collab_model = pickle.load(f)

def get_hybrid_recommendations(input_title, user_id=1, top_n=5):
    if input_title not in df['title'].values:
        return ["Sorry, title not found."]
    idx = df[df['title'] == input_title].index[0]
    sims = cosine_similarity([bert_embeddings[idx]], bert_embeddings).flatten()
    similar_idxs = sims.argsort()[::-1][1:30]

    results = []
    for i in similar_idxs:
        title = df.iloc[i]['title']
        est = collab_model.predict(user_id, title).est
        results.append((title, est))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_n]]
