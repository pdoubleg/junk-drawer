import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
from streamlit_extras.stateful_button import button
from top_n_tool import run_tool
import os
from dotenv import load_dotenv
load_dotenv()
import markdown
import pandas as pd
import numpy as np
import time as time
from tqdm import tqdm
import base64
from pathlib import Path
from typing import List
from io import StringIO
import json
import itertools
import tiktoken
from langchain.embeddings import HuggingFaceInstructEmbeddings
import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".st_langchain.db")

from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.tools.function_tool import FunctionTool

from semantic_search import SemanticSearch
from langchain.chat_models import ChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
import st_utils

from html_templates import css, user_template, bot_template

tqdm.pandas()

# Constants
DATA_PATH = "reddit_legal_cluster_test_results.parquet"

   
@st.cache_data
def get_df():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df['datestamp'] = df['timestamp'].dt.date
    df['State'] = pd.Categorical(df['State'])
    df['text_label'] = pd.Categorical(df['text_label'])
    df['topic_title'] = pd.Categorical(df['topic_title'])
    return df



def get_top_n_report(query: str, top_n: int) -> str:
    """
    Takes a user query and returns a string of top n similar texts.
    """
    return run_tool(query, top_n)



@st.cache_resource
def init_search_engine(df):
    return SemanticSearch(df)


@st.cache_resource
def get_llm(temperature: float = 0, model: str = "gpt-3.5-turbo"):
    llm = ChatOpenAI(temperature=temperature, model=model)
    return llm


def display_description():
    """Displays the description of the app."""
    st.markdown("<h4 style='text-align: left;'>Search a knowledge bank of past questions</h4>", unsafe_allow_html=True)
    st.write(
        """
        Why use this tool?
        - 👉 Find cases relevant to your new question
        - 👉 Get an automated report with citations
        """
    )
    
    
def display_warning():
    """Displays a warning message in small text"""
    st.write(
        "<h8 style='text-align: left;'>⚠️ Warning: This app is currently under construction.</h8>",
        unsafe_allow_html=True,
    )
    
    
def display_known_issues():
    """Displays the known issues"""
    with st.expander("⚠️ Known Issues:", expanded=False):
        st.markdown(
            """
            - 🚨 If search stalls near the end, click the Generate Report button again
            - 🚨 While all sources are cited directly from the data, please verify them
            - 🚨 Citations are accurate, but the generated text might be hallucinated (rare)
            """
        )

def print_response(query: str, response: str) -> None:
    st.markdown(f"## Query:")
    st.markdown(f"{query}")
    st.markdown(f"## Model Response:")
    st.markdown(f"{response}")
    
    
def print_sources(df: pd.DataFrame, response: str) -> None:
    citation_numbers = st_utils.extract_citation_numbers_in_brackets(response)
    st_utils.print_cited_sources(df, citation_numbers)
    
# def print_keywords(df: pd.DataFrame, response: str, keyword_col_name: str) -> None:
#     citation_numbers = st_utils.extract_citation_numbers_in_brackets(response)
#     st_utils.print_keyword_tags(df, citation_numbers, keyword_col_name)


def b_get_feedback():
    if button('Feedback', key="open_feedback"):
        feedback_text = st.text_input("Please provide your feedback")
        feedback_score = st.number_input("Rate your experience (0-10)", min_value=0, max_value=10)
        user_feedback = pd.DataFrame({"Feedback_Text": [feedback_text], "Feedback_Score": [feedback_score]})
        if button('Send', key="send_feedback"):
            if os.path.exists("user_feedback.csv"):
                user_feedback.to_csv("user_feedback.csv", mode='a', header=False, index=False)
            else:
                user_feedback.to_csv("user_feedback.csv", index=False)
            time.sleep(1)
            st.toast("✔️ Feedback received! Thanks for being in the loop 👍\nClick the `Feedback` button to open or close this anytime.")


def app():
    """Main function that runs the Streamlit app."""
    st.markdown(
        "<h2 style='text-align: left;'>GPT Researcher 📚</h2>",
        unsafe_allow_html=True,
    )
    
    # Add a sidebar dropdown for model selection
    model = st.sidebar.selectbox(
        'Select Model',
        ('gpt-3.5-turbo', 'gpt-4')
    )

    load_dotenv()
    df = get_df()
    # # search_engine = init_search_engine(df)
    # llm = get_llm(model=model)
    function_tool = FunctionTool.from_defaults(fn=get_top_n_report)
    llm = OpenAI(model="gpt-3.5-turbo")
    agent = ReActAgent.from_tools([function_tool], llm=llm, verbose=True)
    
    
    # Add a reset button to the sidebar
    reset_button = st.sidebar.button("Reset & Clear")
    if reset_button:
        st.session_state.clear()  # Clear the session state to reset the app
        agent.reset()
    

    display_description()
    # Display sample questions
    with st.expander("❓ Here are some example questions you can ask:", expanded=False):
        st.markdown(
            """
            - Show me questions about getting fired for medical marijuana use while at work and on the job.
            """
        )
    
    st.write(css, unsafe_allow_html=True)
    # Define chat history session state variable
    st.session_state.setdefault('chat_history', [])

    # Get the query from the user and sumit button
    query = st.chat_input(placeholder="Ask me anything!")

    # Add the button to the empty container
    # button = st.button("Generate Report", type='primary')
    # st.dataframe(df[display_cols].head())
    
    if query:
        # with st.status("⏳ Researching past similar cases ...", expanded=True):
        start_time_total = time.time()  # Record the start time
        # answer = run_tool(query, top_n=10, model_name=model)
        answer = agent.chat(query)
        # Store conversation
        st.session_state.chat_history.append(f"{query}")
        st.session_state.chat_history.append(f"{answer}")
        end_time_total = time.time()  # Record the end time
        total_time = end_time_total - start_time_total  # Calculate total runtime
        st.success(f"Research Complete! 🕰️ Total runtime: {total_time:.2f} seconds")
    # Display conversation in reverse order
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0: st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else: st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)


    # display_known_issues()
    # display_warning()
    b_get_feedback()

if __name__ == "__main__":
    app()