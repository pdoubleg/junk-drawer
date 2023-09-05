import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
from streamlit_extras.stateful_button import button
import os
from dotenv import load_dotenv
load_dotenv()

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
import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".st_langchain.db")

from semantic_search import SemanticSearch
from langchain.chat_models import ChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
import st_utils

tqdm.pandas()

# Constants
DATA_PATH = "reddit_legal_cluster_test_results.parquet"

   
@st.cache_data
def get_df():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df['datestamp'] = df['timestamp'].dt.date
    return df


@st.cache_resource
def init_search_engine(df):
    return SemanticSearch(df)


@st.cache_resource
def get_llm(temperature: float = 0, model: str = "gpt-3.5-turbo"):
    llm = ChatOpenAI(temperature=temperature, model=model)
    return llm


def display_description():
    """Displays the description of the app."""
    st.markdown("<h4 style='text-align: left;'>Have AI work for you by rearching a vast knowledge bank of past questions</h4>", unsafe_allow_html=True)
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
        if st.button('Send', key="send_feedback"):
            if os.path.exists("user_feedback.csv"):
                user_feedback.to_csv("user_feedback.csv", mode='a', header=False, index=False)
            else:
                user_feedback.to_csv("user_feedback.csv", index=False)
            time.sleep(1)
            st.toast("✔️ Feedback received! Thank you 👍\nClick `Feedback` to Open or Close this anytime.")



def app():
    """Main function that runs the Streamlit app."""
    st.markdown(
        "<h2 style='text-align: left;'>GPT Researcher 📚</h2>",
        unsafe_allow_html=True,
    )
    
    if "query" not in st.session_state:
        st.session_state.query = []
        
        
    if "response" not in st.session_state:
        st.session_state.response = []
        
    
    # Add a sidebar dropdown for model selection
    model = st.sidebar.selectbox(
        'Select Model',
        ('gpt-3.5-turbo', 'gpt-4')
    )
    
    load_dotenv()
    df = get_df()
    
    # Add sidebar multiselect for filtering
    state_options = ['All'] + df['State'].unique().tolist()
    topic_options = ['All'] + df['topic_title'].unique().tolist()
    states = st.sidebar.multiselect('Select States', options=state_options, default=['All'])
    topics = st.sidebar.multiselect('Select Topics', options=topic_options, default=['All'])
    dates = st.sidebar.multiselect('Select Dates', options=df['datestamp'].unique().tolist())
    top_n = st.sidebar.slider('Select Top N', min_value=1, max_value=50, value=10)

    # Filter the dataframe based on the selected values
    if 'All' not in states:
        df = df[df['State'].isin(states)]
    if 'All' not in topics:
        df = df[df['topic_title'].isin(topics)]
    if dates:
        df = df[df['datestamp'].isin(dates)]     
    
    search_engine = init_search_engine(df)
    llm = get_llm(model=model)

    display_description()
    b_get_feedback()

    # Create a placeholder for the text area
    query_placeholder = st.empty()

    # Get the query from the user and submit button
    query = query_placeholder.text_area(
        "Enter your question here and press Generate Answer:",
        height=200,
        key='query_text_area',
    )

    # Add the buttons to the empty container
    button = st.button("Generate Report", type='primary')
    reset_button = st.button("Clear & Reset", type='secondary')

    if reset_button:
        # Clear the placeholder and recreate the text area
        query_placeholder.empty()
        query = query_placeholder.text_area(
            "Enter your question here and press Generate Answer:",
            height=200,
            key='query_text_area_reset',
        )
        st.experimental_rerun()

    if query and button:
        with st.status("⏳ Getting similar docs ...", expanded=False):
            start_time_total = time.time()  # Record the start time
            
            query_clean = search_engine.preprocess_text(query)
            st.write(query_clean)
                
            df_res = search_engine.query_similar_documents(
                query, # SemanticSearch by default will clean the text
                top_n=top_n,
                filter_criteria=None,
                use_cosine_similarity=True,
                similarity_threshold=0.98,
            )
            st.success(f"Retrieved {df_res.shape[0]} related cases from knowledge bank")
            st.dataframe(df_res[["index", "sim_score", "State", "datestamp", "topic_title", "llm_title", "body"]].head(10))


        with st.status(f"⏳ Generating fact pattern summaries for top-{df_res.shape[0]} cases ...", expanded=False):
            start_time = time.time()
            df_res = st_utils.get_llm_fact_pattern_summary(df=df_res, text_col_name="body")
            st.markdown("#### Example Transformation")
            st.markdown(f"Random sample: index {df_res['index'].sample(1, random_state=42).tolist()[0]}")
            st.markdown("___")
            st.markdown(f"**Original Text:**")
            st.write(df_res['body'].sample(1, random_state=42).tolist()[0])
            st.markdown("___")
            st.markdown("**Revised**:")
            st.write(df_res['summary'].sample(1, random_state=42).tolist()[0])
            elapsed_time = time.time() - start_time
            st.success(f"Analysis complete! 🕰️ Time taken {elapsed_time:.2f} seconds")

            
        with st.status("⏳ Re-ranking search results with cross-encoder...", expanded=True):
            start_time = time.time()
            rerank_res_df = st_utils.rerank_with_cross_encoder(
                df_res, 
                query_clean, 
                'summary')
            
            elapsed_time = time.time() - start_time
            st.success(f"Re-ranking finished! 🕰️ Time taken {elapsed_time:.2f} seconds")
            
        
        # Generate prompt for the model to answer the question
        formatted_input = st_utils.create_formatted_input(
            rerank_res_df,
            query_clean,
        )
        rerank_res_df = st_utils.add_month_year_to_df(rerank_res_df, 'timestamp')

        with st.status("Generating report based on the most similar cases...", expanded=True):
            start_time = time.time()
            response = st_utils.get_final_answer(formatted_input, llm)
            elapsed_time = time.time() - start_time
            st.success(f"Report complete! 🕰️ Time taken {elapsed_time:.2f} seconds")
        
        end_time_total = time.time()  # Record the end time
        total_time = end_time_total - start_time_total  # Calculate total runtime
        st.success(f"🎉 Research Complete! 🕰️ Total runtime: {total_time:.2f} seconds")
        
        print_response(query_clean, response)
        print_sources(rerank_res_df, response)
        # print_keywords(rerank_res_df, response, 'key_phrases_mmr')
        # feedback = streamlit_feedback(
        #     feedback_type="thumbs",
        #     optional_text_label="[Optional] Please provide an explanation",
        #     single_submit=True,
        #     align="flex-start",
        #     key=f"feedback_{start_time_total}",
        #     )
        # dump_logs(query, response, json.dumps(feedback))
        
            
    # Display sample questions
    with st.expander("❓ Here are some example questions you can ask:", expanded=False):
        st.markdown(
            """
            - Show me questions about getting fired for medical marijuana use while at work and on the job.
            """
        )

    # display_known_issues()
    # display_warning()


if __name__ == "__main__":
    app()