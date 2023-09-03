import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
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
from langchain.embeddings import HuggingFaceInstructEmbeddings
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

# Session states
# --------------


   
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


@st.cache_resource
def get_query_embeddings_model():
    query_embeddings_model = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the insurance query for retrieving similar articles: "
    )
    return query_embeddings_model


@st.cache_resource
def get_doc_embeddings_model():
    doc_embeddings_model = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the insurance article for retrieval: "
    )
    return doc_embeddings_model


def display_description():
    """Displays the description of the app."""
    st.markdown("<h4 style='text-align: left;'>Search a knowledge bank of questions</h4>", unsafe_allow_html=True)
    st.write(
        """
        Why use this tool?
        - 👉 Find cases relevant to your new question
        - 👉 Get an automated history report with citations
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
    search_engine = init_search_engine(df)
    llm = get_llm(model=model)
    # query_embeddings_model = get_query_embeddings_model()
    # doc_embeddings_model = get_doc_embeddings_model()

    display_description()

    # Get the query from the user and sumit button
    query = st.text_area(
        "Enter a new question here and press Generate Answer:",
        height=200,
    )

    # Add the button to the empty container
    button = st.button("Generate Report", type='primary')

    if query and button:
        with st.status("⏳ Getting similar docs ...", expanded=True):
            start_time_total = time.time()  # Record the start time
            
            query_clean = search_engine.preprocess_text(query)
            st.write(query_clean)
                
            df_res = search_engine.query_similar_documents(
                query, # SemanticSearch by default will clean the text
                top_n=10,
                filter_criteria=None,
                use_cosine_similarity=True,
                similarity_threshold=0.98,
            )
            st.success(f"Retrieved {df_res.shape[0]} related cases from knowledge bank")
            st.dataframe(df_res[["sim_score", "llm_title", "body"]].head(10))


        
        with st.status(f"⏳ Generating legal fact pattern summaries for top-{df_res.shape[0]} cases ...", expanded=True):
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

            # Calculate the degree to which the row orders changed
            original_order = df_res.index.tolist()
            new_order = rerank_res_df.index.tolist()
            order_changes = sum([i != j for i, j in zip(original_order, new_order)])
            st.write(f"{order_changes} out of {len(df_res)} rows changed order.")

        
        # Generate prompt for the model to answer the question
        formatted_input = st_utils.create_formatted_input(
            rerank_res_df,
            query_clean,
        )
        rerank_res_df = st_utils.add_month_year_to_df(rerank_res_df, 'timestamp')

        with st.status("Generating final report ...", expanded=True):
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