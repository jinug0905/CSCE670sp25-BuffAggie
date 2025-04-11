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

def add_or_update_user(username, gender, major, preferences):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = c.fetchone()

    if exists:
        c.execute(
            "UPDATE users SET gender = ?, major = ?, preferences = ? WHERE username = ?",
            (gender, major, preferences, username)
        )
    else:
        c.execute(
            "INSERT INTO users (username, gender, major, preferences) VALUES (?, ?, ?, ?)",
            (username, gender, major, preferences)
        )

    conn.commit()
    conn.close()

def get_user_info(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username, gender, major, preferences FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row if row else None

def get_all_users():
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

def get_user_preferences(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT preferences FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""


@st.cache_resource(show_spinner="Loading model and dataset...")
def load_model_and_data():
    df = pd.read_csv("megaGymDataset_trimmed.csv")
    df['Desc'] = df['Desc'].fillna("")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['embeddings'] = df['Desc'].apply(lambda x: model.encode(x))
    return df, model


def recommend(query, df, model, top_k=5):
    query_vec = model.encode(query)
    all_embeddings = np.vstack(df['embeddings'].values)
    similarities = cosine_similarity([query_vec], all_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

init_db()
st.set_page_config(page_title="Exercise Recommender", layout="centered")
st.title("🏋️ Exercise Recommender for Aggies")

df, model = load_model_and_data()

# --- Sidebar UI ---
st.sidebar.header("👤 Create or View User Profile")

for key, default in {
    "username": "",
    "gender": "Prefer not to say",
    "major": "",
    "preferences": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

username_widget = st.sidebar.text_input("Username", value=st.session_state['username'], key="username_widget")
gender_widget = st.sidebar.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"],
                 index=["Prefer not to say", "Male", "Female", "Other"].index(st.session_state['gender']), key="gender_widget")
major_widget = st.sidebar.text_input("Major", value=st.session_state['major'], key="major_widget")
prefs_widget = st.sidebar.text_area("Exercise Preferences", value=st.session_state['preferences'], key="prefs_widget")

load_col, save_col = st.sidebar.columns(2)

with load_col:
    if st.button("Load Profile"):
        user_data = get_user_info(username_widget)
        if user_data:
            st.session_state['username'] = user_data[0]
            st.session_state['gender'] = user_data[1]
            st.session_state['major'] = user_data[2]
            st.session_state['preferences'] = user_data[3]
            st.sidebar.success(f"Loaded profile for '{user_data[0]}'.")
            st.rerun()
        else:
            st.sidebar.warning("User not found.")

with save_col:
    if st.button("Save Profile"):
        if username_widget.strip() == "":
            st.sidebar.warning("Please enter a username.")
        else:
            add_or_update_user(username_widget, gender_widget, major_widget, prefs_widget)
            st.sidebar.success(f"Profile for '{username_widget}' saved or updated!")

st.subheader("🔍 Get Exercise Recommendations")
query = st.text_input("💬 Describe your workout goal (e.g., 'build upper body strength'):")

if st.button("Recommend Exercises"):
    if st.session_state['username'].strip() == "":
        st.warning("Please enter your username in the sidebar to personalize recommendations.")
    elif query.strip() == "":
        st.warning("Please enter a workout goal.")
    else:
        user_pref = get_user_preferences(st.session_state['username'])
        full_query = f"{user_pref} {query}" if user_pref else query

        results = recommend(full_query, df, model)
        st.subheader("🔥 Top Matches:")
        for _, row in results.iterrows():
            st.markdown(f"**{row['Title']}**")
            st.write(row['Desc'])
            st.caption(f"Similarity Score: {row['similarity']:.2f}")
