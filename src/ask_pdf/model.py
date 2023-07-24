import streamlit as st
import datetime
from collections import Counter
from time import time as now
import hashlib
import re
import io
import os
import spacy
import pdf
import ai
from llama_index import (
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage
)
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


def get_llm(model_temperature):
    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(temperature=model_temperature, model_name="gpt-3.5-turbo-0613")


def pdf_to_pages(file):
    "extract text (pages) from pdf file"
    pdf = pypdf.PdfReader(file)
    pages = [page.extract_text() for page in pdf.pages]
    return pages

# helpers
def remove_page_numbers(text):
    return re.sub(r'Page \d+ of \d+', '', text)

nlp = spacy.load('en_core_web_lg')

def find_eos_spacy(text, nlp):
    doc = nlp(text)
    return [sent.end_char for sent in doc.sents]

def fix_text_problems(text):
    # Strip footer and watermark
    watermarks = [
        "SAMPLE", 
        "HO 00 03 10 00 Copyright, Insurance Services Office, Inc., 1999", 
        "HOMEOWNERS 3 – SPECIAL FORM", 
        "HOMEOWNERS", 
        "HO 00 03 10 00", 
        "Copyright, Insurance Services Office, Inc., 1999",
    ]

    for watermark in watermarks:
        text = text.replace(watermark, '')

    text = remove_page_numbers(text)

    # Merge hyphenated words
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    # Add a space after a period or a colon if it's not there yet
    text = re.sub(r'([.:])([^ \n])', r'\1 \2', text)
    # Add a space after a closing parenthesis if it's not there yet
    text = re.sub(r'(\))(?=[^\s])', r'\1 ', text)
    # Add a space before an opening parenthesis if it's not there yet
    text = re.sub(r'(?<=[^\s])(\()', r' \1', text)
    # Normalize whitespaces, but keep newlines
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text

def use_key(api_key):
	ai.use_key(api_key)

def set_user(user):
	ai.set_user(user)

@st.cache_resource
def initialize_policy_index():
    llm = get_llm(model_temperature=0)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults(persist_dir="./storage_v2")

    ho3_index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context
        )
    return ho3_index

index = initialize_policy_index()

@st.cache_resource
def get_basic_retriever(_index):
    query_engine = _index.as_query_engine()
    return query_engine

ho3_query_engine = get_basic_retriever(index)

def query(text):
	out = ho3_query_engine.query(text)
	return out
