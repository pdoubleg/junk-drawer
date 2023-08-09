import base64
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List
import pandas as pd
from io import BytesIO

import openai
# LangChain
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document as LC_Document
# Llama-Index
from llama_index import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage
)
from llama_index.schema import Document as LI_Document

from ho3_sample_policy_index import build_ho3_sample_policy_index
from ho3_sub_query_engine import build_ho3_sub_query_engine

from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv

# set to DEBUG for more verbose logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    openai.api_key = os.getenv("OPENAI_API_KEY")

   
# ------------------- index creation ------------------- #

def parse_pdf(file: BytesIO):

    pdf = PdfReader(file)
    text_list = []

    # Get the number of pages in the PDF document
    num_pages = len(pdf.pages)

    # Iterate over every page
    for page in range(num_pages):
        # Extract the text from the page
        page_text = pdf.pages[page].extract_text()
        text_list.append(page_text)

    text = "\n".join(text_list)

    return [LI_Document(text)]


# def create_index(pdf_obj, folder_name=None, file_name=None):
#     """
#     Create an index for a given PDF file and upload it to S3.
#     """
#     index_name = file_name.replace(".pdf", ".json")

#     logging.info("Generating new index...")
#     documents = parse_pdf(pdf_obj)

#     logging.info("Creating index...")
#     index = VectorStoreIndex(documents)

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_path = f"{tmp_dir}/{index_name}"
#         logging.info("Saving index...")
#         index.save_to_disk(tmp_path)

#         # with open(tmp_path, "rb") as f:
#         #     logging.info("Uploading index to s3...")
#         #     s3.upload_files(f, f"{folder_name}/{index_name}")

#     return index


@st.cache_resource(show_spinner=False)
def get_index(chosen_index):
    """
    Get the index
    """
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="../_policy_index_storage"))
    # if s3.file_exists(folder_name, index_name):
    #     logging.info("Index found, loading index...")
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         tmp_path = f"{tmp_dir}/{index_name}"
    #         s3.download_file(f"{folder_name}/{index_name}", tmp_path)
    #         index = VectorStoreIndex.load_from_disk(tmp_path)

    # else:
        # logging.info("Index not found, generating index...")
        # with tempfile.NamedTemporaryFile("wb") as f_src:
        #     logging.info(f"{file_name} downloaded")
        #     # s3.download_file(f"{folder_name}/{file_name}", f_src.name)

        #     with open(f_src.name, "rb") as f:
        #         index = create_index(f, folder_name, file_name)

    return index


def query_gpt(chosen_index="../_policy_index_storage", query=None):
    
    if not os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # LLM Predictor (gpt-3.5-turbo)
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
    )

    index = get_index(chosen_index)
    response = index.as_query_engine().query(query, llm=llm)

    logging.info(response.get_formatted_sources())

    return response


@st.cache_resource
def create_tool(_index):
    tools = [
        Tool(
            name="policy index",
            func=lambda q: str(_index.as_query_engine().query(q)),
            description=f"Useful to answering questions about insurance policy coverage",
            return_direct=True,
        ),
    ]

    return tools


@st.cache_resource
def create_agent(chosen_index="../_policy_index_storage"):
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    index = get_index(chosen_index)
    tools = create_tool(index)

    agent = initialize_agent(
        tools, llm, agent="conversational-react-description", memory=memory
    )

    return agent

def query_gpt_memory(chosen_index, query):

    agent = create_agent(chosen_index)
    res = ""

    try:
        res = agent.run(input=query)
    except Exception as e:
        logging.error(e)
        res = "Something went wrong... Please try again."

    st.session_state.memory = agent.memory.buffer

    return res


# ------------------- styling ------------------- #


custom_css = """
    <style>
    .card {
        padding: 20px;
        border: 1px solid #ccc;
        border-radius: 10px;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .card-content {
        font-size: 16px;
        color: white;
    }
    </style>
    """


def create_custom_markdown_card(text):
    # Apply custom CSS styles to the card
    st.markdown(custom_css, unsafe_allow_html=True)
    # Create the card
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Information</div>
            <div class="card-content">
            """
        + text
        + """
        
        </div>
            """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")
    st.write("")
    
    
def wrap_doc_in_html(docs: List[str]) -> str:
    """Wraps each page in document separated by newlines in <p> tags"""
    # Convert to langchain schema
    docs_ = [LI_Document.to_langchain_format(doc) for doc in docs]
    text = [doc.page_content for doc in docs_]
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


# ------------------- Render PDF ------------------- #

# New version
@st.cache_data
def show_pdf(file_name):

    with tempfile.NamedTemporaryFile("wb") as f_src:
        logging.info(f"Downloading {file_name}...")
        # s3.download_file(f"{folder_name}/{file_name}", f_src.name)

        with open(file_name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%" height="1000"
            type="application/pdf"
            style="min-width: 400px;"
        >
        </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)
        

# Original function
@st.cache_data
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# ------------------- LangChain Version ------------------- #


def get_lc_docs_from_df(df: pd.DataFrame, 
                        text_col: str, 
                        **kwargs: Dict[str, Any]):
    """
    Get a list of LangChain Document objects from a dataframe
    
    Example:
    # Input df and column containing "document text"
    df = ...
    text_col = 'text_col_name'
    # Prepare metadata dictionary
    metadata = {'source': 'doc_id', 
                'category': 'policy document', 
                'coverage_line': 'property'}
    # Apply the function
    docs = get_lc_docs_from_df(df, text_col, **metadata)
    """
    docs = []
    for i, row in df.iterrows():
        metadata = {key: row[value] for key, value in kwargs.items()}
        doc = LC_Document(
            page_content=row[text_col],
            metadata=metadata,
            lookup_index=i,
        )
        docs.append(doc)
    return docs


def get_li_docs_from_df(df: pd.DataFrame, 
                        text_col: str, 
                        exclude_from_llm: List[str], 
                        exclude_from_embed: List[str],  
                        **kwargs):
    """
    Get a list of LangChain Document objects from a dataframe
    
    Example:
    # Input df and column containing "document text"
    df = ...
    text_col = 'text_col_name'
    # Prepare metadata dictionary
    metadata = {'source': 'doc_id', 
                'category': 'policy document', 
                'coverage_line': 'property'}
    # Apply the function
    docs = get_lc_docs_from_df(df, text_col, **metadata)
    """
    docs = []
    for i, row in df.iterrows():
        metadata = {key: row[value] for key, value in kwargs.items()}
        doc = LI_Document(
            text=row[text_col],
            id_=i,
            metadata=metadata,
            excluded_llm_metadata_keys=exclude_from_llm,
            excluded_embed_metadata_keys=exclude_from_embed,
        )
        docs.append(doc)
    return docs





