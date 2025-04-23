import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from typing_extensions import TypedDict
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

### --- Database Setup --- ###
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            gender TEXT,
            major TEXT,
            preferences TEXT, 
            gym TEXT
        )
    ''')

    # Ratings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            exercise TEXT,
            rating INTEGER CHECK(rating BETWEEN 1 AND 5),
            UNIQUE(username, exercise)
        )
    ''')
    conn.commit()
    conn.close()

# User Profile DB functions
def add_or_update_user(username, gender, major, preferences):
    gym = major_to_gym.get(major, "Main Rec")
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = c.fetchone()

    if exists:
        c.execute(
            "UPDATE users SET gender = ?, major = ?, preferences = ?, gym = ? WHERE username = ?",
            (gender, major, preferences, gym, username)
        )
    else:
        c.execute(
            "INSERT INTO users (username, gender, major, preferences, gym) VALUES (?, ?, ?, ?, ?)",
            (username, gender, major, preferences, gym)
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
    df['combined_text'] = df.apply(lambda row: f"{row['Title']}. {row['Desc']}", axis=1)
    df['embeddings'] = df['combined_text'].apply(lambda x: model.encode(x))
    return df, model

def recommend(query, df, model, top_k=5):
    query_vec = model.encode(query)
    all_embeddings = np.vstack(df['embeddings'].values)
    similarities = cosine_similarity([query_vec], all_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

# User ratings DB functions
def submit_rating(username, exercise, rating):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO ratings (username, exercise, rating)
        VALUES (?, ?, ?)
    ''', (username, exercise, rating))
    conn.commit()
    conn.close()

def get_average_rating(exercise):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        SELECT AVG(rating) FROM ratings WHERE exercise = ?
    ''', (exercise,))
    avg = c.fetchone()[0]
    conn.close()
    return round(avg, 2) if avg else "No ratings yet."

def get_user_rating(username, exercise):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        SELECT rating FROM ratings WHERE username = ? AND exercise = ?
    ''', (username, exercise))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0


### Langgraph Integration ###
class Intent(BaseModel):
    """Classify user intent."""
    intent: Literal["workout", "avoid_workout"] = Field(..., description="Classify the user query as a good workout query or asking for a workout that is opposite of another / avoids a body part.")

groq_api_key = os.getenv("GROQ_API_KEY", "NO_KEY_FOUND")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.3-70b-Versatile")

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at classifying user queries. Classify the following query as 'workout' or 'avoid_workout'."),
    ("human", "{query}")
])

intent_classifier = intent_prompt | llm.with_structured_output(Intent)

def transform_query(state):
    """Modify non-workout queries into contextually appropriate workout-related queries."""
    print("Transforming Node called")
    query = state["query"]
    intent = intent_classifier.invoke({"query": query}).intent
    print("Transforming Node called with intent:", intent)
    if intent == "avoid_workout":
        # Use the LLM to modify the query to focus on appropriate workouts
        transformed_query = llm.invoke(
            [HumanMessage(content=f"""Given the query '{query}', suggest a workout-related query that avoids the injured body part or focuses on alternative exercises. These queries must be concise, maximum of 10 words and use opposite body part keywords and not include the names of the body parts the user wants to avoid
    Examples: If Given 'I injured my arms, give me exercises', return 'Give me Quadriceps, Abdominals, Hamstrings, Calves, or other lower body exercises.'
    Examples: If Given 'Give me alternates to lower body', return 'Give me  upper body exercises.'""")]
        )
        print("Transformed query:", transformed_query.content)
        return {"query": transformed_query.content}  # Extract the content from the response
    return {"query": query}

def recommend_workouts(state):
    """Recommend workouts based on the query."""
    print("Recommendation Node called")
    query = state["query"]
    results = recommend(query, state["df"], state["model"])
    return {"results": results}

class WorkflowState(TypedDict):
    query: str
    df: pd.DataFrame
    model: SentenceTransformer
    results: pd.DataFrame

# Langgraph Workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("transform_query", transform_query)
workflow.add_node("recommend_workouts", recommend_workouts)
workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "recommend_workouts")
workflow.add_edge("recommend_workouts", END)


##############################
#
# Recommendation system
#
##############################

@st.cache_resource(show_spinner="Generating CF recommendations...")
def generate_item_item_cf():
    conn = sqlite3.connect("users.db")
    ratings_df = pd.read_sql("SELECT username, exercise, rating FROM ratings", conn)
    conn.close()

    # Pivot table to get item-user rating matrix
    item_user_matrix = ratings_df.pivot_table(
        index='exercise',
        columns='username',
        values='rating'
    ).fillna(0)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(item_user_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, 
                                 index=item_user_matrix.index, 
                                 columns=item_user_matrix.index)
    return similarity_df, ratings_df

def recommend_cf_exercises(username, similarity_df, ratings_df, top_n=5):
    user_ratings = ratings_df[ratings_df['username'] == username]
    rated_exercises = user_ratings['exercise'].tolist()

    scores = {}
    for exercise in rated_exercises:
        rating = user_ratings.loc[user_ratings['exercise'] == exercise, 'rating'].values[0]
        similar_items = similarity_df[exercise]

        for item, similarity in similar_items.items():
            if item not in rated_exercises:
                scores[item] = scores.get(item, 0) + similarity * rating

    recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommended_items

# Fall back for new users
def get_popular_exercises(ratings_df, top_n=5):
    popular = ratings_df.groupby('exercise')['rating'].mean().sort_values(ascending=False).head(top_n)
    return list(popular.items())

##############################
#
# Streamlit App
#
##############################
init_db()

st.set_page_config(page_title="Exercise Recommender", layout="centered")
st.title("Exercise Recommender for Aggies")

df, model = load_model_and_data()

major_to_gym = {
    "Prefer not to say": "Main Rec",
    "Engineering/CS": "Polo",
    "Math/Science": "Polo",
    "Liberal Arts": "Southside",
    "Humanities": "Southside",
    "Education": "Southside",
    "Fine Arts": "Southside",
    "Business": "Main Rec",
    "Social Science": "Main Rec",
    "Health": "Main Rec",
    "Other": "Main Rec"
}

### --- Sidebar UI --- ###
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
major_options = [
    "Prefer not to say",
    "Business",
    "Education",
    "Engineering/CS",
    "Fine Arts",
    "Health",
    "Humanities",
    "Liberal Arts",
    "Math/Science",
    "Social Science",
    "Other"
]

major_widget = st.sidebar.selectbox(
    "Major",
    options=major_options,
    index=major_options.index(st.session_state['major']) if st.session_state['major'] in major_options else 0,
    key="major_widget"
)

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


### --- Page Body --- ###
# st.subheader("🔍 Get Exercise Recommendations")
query = st.text_input("💬 Describe your workout goal (e.g., 'build upper body strength'):")

def generate_summary(exercise_names):
    """Generate a workout plan summary using the LLM."""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at summarizing workout plans. Summarize the following exercises into a concise workout plan which includes sets and reps:"),
        ("human", "{exercise_names}")
    ])
    summary_llm = summary_prompt | llm
    summary = summary_llm.invoke({"exercise_names": "\n".join(exercise_names)})
    return summary.content
# Replace your current Streamlit display logic with this snippet

if 'recommendation_generated' not in st.session_state:
    st.session_state['recommendation_generated'] = False

if st.button("Recommend Exercises"):
    if st.session_state['username'].strip() == "":
        st.warning("Please enter your username in the sidebar to personalize recommendations.")
    elif query.strip() == "":
        st.warning("Please enter a workout goal.")
    else:
        user_pref = get_user_preferences(st.session_state['username'])
        full_query = query.replace("only", "").strip() if "only" in query.lower() else f"{user_pref} {query}" if user_pref else query

        state = {"query": full_query, "df": df, "model": model}
        result_state = workflow.compile().invoke(state)
        results = result_state["results"]

        user_major = st.session_state.get('major', '')
        user_gym = major_to_gym.get(user_major, "Main Rec")
        
        # Store results into session state for dropdown display
        st.session_state['location_rec'] = (
            f"🎓 As a general Aggie student, the **Student Rec Center** is the nicest gym on campus, and the perfect place for you to train!"
            if user_major == "Other" or user_major == "Prefer not to say"
            else f"As a **{user_major}** major, the **{user_gym} Rec Center** would be the perfect place for you to train!"
        )

        st.session_state['top_matches'] = []
        exercise_names = []
        for _, row in results.iterrows():
            match_text = f"**{row['Title']}**\n\n{row['Desc']}\n\n_Similarity Score: {row['similarity']:.2f}_"
            st.session_state['top_matches'].append(match_text)
            exercise_names.append(row['Title'])

        if exercise_names:
            summary = generate_summary(exercise_names)
            st.session_state['workout_summary'] = summary
        else:
            st.session_state['workout_summary'] = "No suitable exercises found."

        st.session_state['recommendation_generated'] = True
        st.rerun()

# Utility function for star rating UI
def star_rating_widget(key, current_rating):
    return st.radio(
        "Your Rating:",
        ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        index=current_rating - 1 if current_rating else None,
        key=key,
        horizontal=True
    )


# Recommendation system implementation
similarity_df, ratings_df = generate_item_item_cf()
username = st.session_state.get('username', '').strip()

# Function to fetch exercise description
def get_exercise_description(title, exercise_df):
    desc_row = exercise_df[exercise_df['Title'] == title]
    return desc_row['Desc'].values[0] if not desc_row.empty else "No description available."

# Updated CF Recommendation block
if username:
    with st.expander("✨ **Recommended Exercises (CF)**", expanded=True):
        recommended_exercises = recommend_cf_exercises(username, similarity_df, ratings_df)

        if recommended_exercises:
            for exercise, score in recommended_exercises:
                avg_rating = get_average_rating(exercise)
                description = get_exercise_description(exercise, df)

                st.markdown(f"### {exercise}")
                st.markdown(description)
                st.caption(f"_Score: {score:.2f} | Global Avg Rating: {avg_rating}_")
                st.markdown("---")
        else:
            st.info("You haven't rated exercises yet, so here are some popular ones to get you started:")
            popular_exercises = get_popular_exercises(ratings_df, top_n=5)
            for exercise, avg_rating in popular_exercises:
                description = get_exercise_description(exercise, df)

                st.markdown(f"### {exercise}")
                st.markdown(description)
                st.caption(f"_Global Avg Rating: {avg_rating:.2f}_")
                st.markdown("---")
else:
    with st.expander("✨ **Recommended Exercises (CF)**", expanded=True):
        st.info("Log in to see personalized recommendations based on your ratings!")


# Display dropdowns if a recommendation was generated
if st.session_state['recommendation_generated']:
    with st.expander("📍 **Recommended Gym Location**", expanded=True):
        st.markdown(st.session_state['location_rec'])

    with st.expander("🔥 **Top Exercise Matches from Query**", expanded=False):
        username = st.session_state.get('username', '').strip()

        for i, match in enumerate(st.session_state['top_matches']):
            exercise_title = match.split('\n')[0].strip("**")
            st.markdown(match)

            col1, col2 = st.columns([1, 2])

            # Display global average rating
            avg_rating = get_average_rating(exercise_title)
            col1.write(f"🌎 Global Avg: {avg_rating}")

            # Star rating UI
            if username:
                user_current_rating = get_user_rating(username, exercise_title)
                user_rating = star_rating_widget(f"{exercise_title}_rating_{i}", user_current_rating)

                if st.button(f"Submit rating for '{exercise_title}'", key=f"{exercise_title}_submit_{i}"):
                    numeric_rating = len(user_rating)
                    submit_rating(username, exercise_title, numeric_rating)
                    st.success(f"Your rating of {numeric_rating} stars submitted for '{exercise_title}'!")
                    st.rerun()
            else:
                col2.info("Log in to rate exercises.")
            
            st.markdown("---")

    with st.expander("📋 **Workout Plan Summary**", expanded=True):
        st.markdown(st.session_state['workout_summary'])
else:
    # Show empty dropdown placeholders initially
    with st.expander("📍 **Recommended Gym Location**", expanded=True):
        st.write("Recommendations will appear here after your query.")

    with st.expander("🔥 **Top Exercise Matches**", expanded=True):
        st.write("Exercise matches will appear here after your query.")

    with st.expander("📋 **Workout Plan Summary**", expanded=True):
        st.write("Workout summary will appear here after your query.")

