
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix Recommender", layout="centered")

@st.cache_resource
def load_data():
    df = pd.read_csv("netflix_clean.csv")
    with open("bert_embeddings.pkl", "rb") as f:
        bert_embeddings = pickle.load(f)
    with open("collaborative_model.pkl", "rb") as f:
        collab_model = pickle.load(f)
    return df, bert_embeddings, collab_model

df, bert_embeddings, collab_model = load_data()
titles = df['title'].tolist()

st.markdown(
    '''
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .recommendation-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎬 Netflix Recommendation Bot</div>', unsafe_allow_html=True)
st.write("Get personalized Netflix recommendations using hybrid AI (BERT + SVD).")

selected_title = st.selectbox("🔍 Start typing a Netflix title:", titles)

# Genre filter
available_genres = sorted(set(genre.strip() for g in df['listed_in'].dropna() for genre in g.split(',')))
selected_genres = st.multiselect("🎭 Filter by Genre", available_genres)

# Language filter
available_languages = sorted(df['language'].dropna().unique())
selected_languages = st.multiselect("🌐 Filter by Language", available_languages)

# Release year filter
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
selected_year = st.slider("📅 Filter by Release Year", min_year, max_year, (min_year, max_year))

def get_hybrid_recommendations(input_title, user_id=1, top_n=5):
    if input_title not in df['title'].values:
        return ["Title not found."]
    idx = df[df['title'] == input_title].index[0]
    sims = cosine_similarity([bert_embeddings[idx]], bert_embeddings).flatten()
    similar_idxs = sims.argsort()[::-1][1:100]

    results = []
    for i in similar_idxs:
        row = df.iloc[i]
        title = row['title']
        genres = row.get('listed_in', '')
        language = row.get('language', '')
        year = row.get('release_year', 0)

        if selected_genres and not any(g.strip() in genres for g in selected_genres):
            continue
        if selected_languages and language not in selected_languages:
            continue
        if not (selected_year[0] <= int(year) <= selected_year[1]):
            continue

        est = collab_model.predict(user_id, title).est
        results.append((title, est))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_n]

if st.button("Recommend"):
    with st.spinner("Crunching recommendations with hybrid AI..."):
        recs = get_hybrid_recommendations(selected_title, user_id=1, top_n=5)

    st.subheader("🎯 Top Recommendations")
    for title, score in recs:
        st.markdown(f'''
        <div class="recommendation-card">
            <h4>{title}</h4>
            <p>Predicted Interest Score: ⭐ {score:.2f}</p>
        </div>
        ''', unsafe_allow_html=True)
