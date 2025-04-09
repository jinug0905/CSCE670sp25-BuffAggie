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

# Load Data + Model (cached for performance)
@st.cache_resource(show_spinner="Loading model and dataset...")
def load_model_and_data():
    df = pd.read_csv("megaGymDataset_trimmed.csv")
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

# Langgraph Integration
class Intent(BaseModel):
    """Classify user intent."""
    intent: Literal["workout", "avoid_workout"] = Field(..., description="Classify the user query as a good workout query or asking for a workout that is opposite of another / avoids a body part.")

llm = ChatGroq(groq_api_key="REPLACE WITH YOUR KEY", model_name="Llama-3.3-70b-Versatile")

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at classifying user queries. Classify the following query as 'workout' or 'avoid_workout'."),
    ("human", "{query}")
])

intent_classifier = intent_prompt | llm.with_structured_output(Intent)

def transform_query(state):
    """Modify non-workout queries into contextually appropriate workout-related queries.
    These queries must be concise, maximum of 10 words and use opposite body part keywords and not include the names of the body parts the user wants to avoid
    Examples: If Given 'I injured my arms, give me exercises', return 'Give me Quadriceps, Abdominals, Hamstrings, or Calves exercises.'
    Examples: If Given 'Give me alternates to lower body', return 'Give me Quadriceps, Shoulders, Lats, Biceps, Forearms, or other upper body exercises.'
    """
    print("Transforming Node called")
    query = state["query"]
    intent = intent_classifier.invoke({"query": query}).intent
    print("Transforming Node called with intent:", intent)
    if intent == "avoid_workout":
        # Use the LLM to modify the query to focus on appropriate workouts
        transformed_query = llm.invoke(
            [HumanMessage(content=f"""Given the query '{query}', suggest a workout-related query that avoids the injured body part or focuses on alternative exercises. These queries must be concise, maximum of 10 words and use opposite body part keywords and not include the names of the body parts the user wants to avoid
    Examples: If Given 'I injured my arms, give me exercises', return 'Give me Quadriceps, Abdominals, Hamstrings, or Calves exercises.'
    Examples: If Given 'Give me alternates to lower body', return 'Give me Quadriceps, Shoulders, Lats, Biceps, Forearms, or other upper body exercises.'""")]
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

# Streamlit App
st.set_page_config(page_title="Exercise Recommender", layout="centered")
st.title("🏋️ Exercise Recommender")

df, model = load_model_and_data()

query = st.text_input("💬 Describe your workout goal (e.g., 'build upper body strength'):")

if st.button("🔍 Get Recommendations") and query:
    state = {"query": query, "df": df, "model": model}
    result_state = workflow.compile().invoke(state)
    results = result_state["results"]

    st.subheader("🔥 Top Matches:")
    for _, row in results.iterrows():
        st.markdown(f"**{row['Title']}**")
        st.write(row['Desc'])
        st.caption(f"Similarity Score: {row['similarity']:.2f}")
