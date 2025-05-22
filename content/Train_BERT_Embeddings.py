import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/netflix_titles.csv").dropna(subset=["description"])
df['title'] = df['title'].str.strip()
descriptions = df["description"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(descriptions, show_progress_bar=True)

with open("bert_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

df.to_csv("netflix_clean.csv", index=False)
print("✅ bert_embeddings.pkl and netflix_clean.csv saved!")
