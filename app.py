import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data + Model (cached for performance)
@st.cache_resource(show_spinner="Loading model and dataset...")
def load_model_and_data():
    df = pd.read_csv("ver_03\megaGymDataset_trimmed.csv")
    df['Desc'] = df['Desc'].fillna("")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['embeddings'] = df['Desc'].apply(lambda x: model.encode(x))
    return df, model

# Recommender Function
def recommend(query, df, model, top_k=5):
    query_vec = model.encode(query)
    all_embeddings = np.vstack(df['embeddings'].values)
    similarities = cosine_similarity([query_vec], all_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

# Streamlit App
st.set_page_config(page_title="Exercise Recommender", layout="centered")
st.title("🏋️ Exercise Recommender")

df, model = load_model_and_data()

query = st.text_input("💬 Describe your workout goal (e.g., 'build upper body strength'):")

if st.button("🔍 Get Recommendations") and query:
    results = recommend(query, df, model)
    st.subheader("🔥 Top Matches:")
    for _, row in results.iterrows():
        st.markdown(f"**{row['Title']}**")
        st.write(row['Desc'])
        st.caption(f"Similarity Score: {row['similarity']:.2f}")
