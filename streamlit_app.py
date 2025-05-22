
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import openai
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Recommender", layout="centered")

openai.api_key = "{openai_api_key}"
OMDB_API_KEY = "519fa74f"

@st.cache_resource
def load_data():
    df = pd.read_csv("netflix_clean.csv")
    with open("bert_embeddings.pkl", "rb") as f:
        bert_embeddings = pickle.load(f)
    with open("collaborative_model.pkl", "rb") as f:
        collab_model = pickle.load(f)
    return df, bert_embeddings, collab_model

df, bert_embeddings, collab_model = load_data()
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df['duration_min'] = df['duration'].str.extract(r'(\d+)').astype(float)
titles = df['title'].tolist()

def fetch_omdb_data(title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
    try:
        response = requests.get(url)
        data = response.json()
        return {
            'poster': data.get('Poster', ''),
            'rating': data.get('imdbRating', 'N/A'),
            'year': data.get('Year', 'N/A')
        }
    except:
        return {'poster': '', 'rating': 'N/A', 'year': 'N/A'}

def parse_query(query):
    prompt = f"""
    Extract the following from this movie recommendation query:
    - reference title (optional)
    - genre (optional)
    - year range (optional, e.g., 2010-2020)
    - max runtime in minutes (optional)

    Query: '{query}'
    Return as JSON.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response['choices'][0]['message']['content']
        return eval(content)
    except Exception as e:
        st.error(f"LLM failed: {e}")
        return {}

def get_recommendations(query, sort_by="score"):
    params = parse_query(query)
    ref_title = params.get("reference title", "")
    genre = params.get("genre", "").lower()
    year_range = params.get("year range", "")
    max_runtime = params.get("max runtime in minutes", None)

    results = []
    if ref_title and ref_title in df['title'].values:
        idx = df[df['title'] == ref_title].index[0]
        sims = cosine_similarity([bert_embeddings[idx]], bert_embeddings).flatten()
        similar_idxs = sims.argsort()[::-1][1:100]
    else:
        sims = cosine_similarity(bert_embeddings, bert_embeddings)
        similar_idxs = np.argsort(-sims.mean(axis=0))[:100]

    for i in similar_idxs:
        row = df.iloc[i]
        title = row['title']
        genres = row['listed_in'].lower() if pd.notnull(row['listed_in']) else ""
        year = row['release_year']
        duration = row['duration_min']

        if genre and genre not in genres:
            continue
        if year_range:
            try:
                start, end = map(int, year_range.split('-'))
                if not (start <= year <= end):
                    continue
            except:
                pass
        if max_runtime and duration and duration > max_runtime:
            continue

        est = collab_model.predict(1, title).est
        results.append((title, est))

    if sort_by == "rating":
        results = sorted(results, key=lambda x: float(fetch_omdb_data(x[0])['rating'] or 0), reverse=True)
    elif sort_by == "year":
        results = sorted(results, key=lambda x: int(fetch_omdb_data(x[0])['year'] or 0), reverse=True)
    else:
        results = sorted(results, key=lambda x: x[1], reverse=True)

    return results[:5]

st.title("🎬 Netflix AI Recommendation Bot")
query = st.text_input("🧠 Describe what you're in the mood to watch:")
sort_by = st.selectbox("🔽 Sort by:", ["score", "rating", "year"])

if st.button("Get Recommendations"):
    with st.spinner("Asking the AI..."):
        recs = get_recommendations(query, sort_by=sort_by)

    for title, score in recs:
        info = fetch_omdb_data(title)
        st.image(info["poster"], width=160)
        st.markdown(f"**{title}**")
        st.markdown(f"⭐ IMDb Rating: {info['rating']} | 📅 {info['year']}")
        st.markdown(f"🔮 Interest Score: {score:.2f}")
        st.markdown("---")
