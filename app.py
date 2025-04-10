import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            gender TEXT,
            major TEXT,
            preferences TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, gender, major, preferences):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, gender, major, preferences) VALUES (?, ?, ?, ?)",
              (username, gender, major, preferences))
    conn.commit()
    conn.close()

def get_user_preferences(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT preferences FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""


# --- Load Model and Dataset ---
@st.cache_resource(show_spinner="Loading model and dataset...")
def load_model_and_data():
    df = pd.read_csv("megaGymDataset_trimmed.csv")
    df['Desc'] = df['Desc'].fillna("")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['embeddings'] = df['Desc'].apply(lambda x: model.encode(x))
    return df, model

# --- Recommender ---
def recommend(query, df, model, top_k=5):
    query_vec = model.encode(query)
    all_embeddings = np.vstack(df['embeddings'].values)
    similarities = cosine_similarity([query_vec], all_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

# --- Initialize DB ---
init_db()

# --- Streamlit UI ---
st.set_page_config(page_title="Exercise Recommender", layout="centered")
st.title("🏋️ Exercise Recommender for Aggies")

df, model = load_model_and_data()

# --- Sidebar: User Profile ---
st.sidebar.header("👤 Create or View User Profile")

with st.sidebar.form("user_form"):
    username = st.text_input("Username")
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    major = st.text_input("Major")
    preferences = st.text_area("Exercise Preferences (e.g., 'cardio, strength, flexibility')")

    submitted = st.form_submit_button("Save Profile")
    if submitted:
        if username.strip() == "":
            st.sidebar.warning("Please enter a username.")
        else:
            try:
                add_user(username, gender, major, preferences)
                st.sidebar.success(f"Profile for '{username}' created!")
            except sqlite3.IntegrityError:
                st.sidebar.warning("Username already exists.")

# --- Main App: Recommendation Section ---
st.subheader("🔍 Get Exercise Recommendations")

query = st.text_input("💬 Describe your workout goal (e.g., 'build upper body strength'):")

if st.button("Recommend Exercises"):
    if username.strip() == "":
        st.warning("Please enter your username in the sidebar to personalize recommendations.")
    elif query.strip() == "":
        st.warning("Please enter a workout goal.")
    else:
        user_pref = get_user_preferences(username)
        full_query = f"{user_pref} {query}" if user_pref else query # attach user pref on top of query

        results = recommend(full_query, df, model)
        st.subheader("🔥 Top Matches:")
        for _, row in results.iterrows():
            st.markdown(f"**{row['Title']}**")
            st.write(row['Desc'])
            st.caption(f"Similarity Score: {row['similarity']:.2f}")



#=-=-=-=-=-=-=-=-=-=-=-=-
# Old Streamlit implementation

# st.set_page_config(page_title="Exercise Recommender", layout="centered")
# st.title("🏋️ Exercise Recommender")

# df, model = load_model_and_data()

# query = st.text_input("💬 Describe your workout goal (e.g., 'build upper body strength'):")

# if st.button("🔍 Get Recommendations") and query:
#     results = recommend(query, df, model)
#     st.subheader("🔥 Top Matches:")
#     for _, row in results.iterrows():
#         st.markdown(f"**{row['Title']}**")
#         st.write(row['Desc'])
#         st.caption(f"Similarity Score: {row['similarity']:.2f}")
