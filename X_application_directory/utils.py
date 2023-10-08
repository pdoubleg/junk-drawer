import datetime
import itertools
import os
import re
from io import BytesIO
from typing import List

import spacy
import tqdm
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
from collections import defaultdict

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.embeddings import LangchainEmbedding


def print_message(*s, condition=True, pad=False, sep=None):
    s = " ".join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f"\n{msg}\n"
        print(msg, flush=True, sep=sep)

    return msg


def create_directory(path):
    if os.path.exists(path):
        print("\n")
        print_message("#> Note: Output directory", path, "already exists\n\n")
    else:
        print("\n")
        print_message("#> Creating directory", path, "\n\n")
        os.makedirs(path)


def deduplicate(seq: list[str]) -> list[str]:
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result


def int_or_float(val):
    if "." in val:
        return float(val)

    return int(val)


def get_llm(temperature):
    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo-0613")


def get_llama_embeddings_model():
    return LangchainEmbedding(OpenAIEmbeddings())


def get_lc_embeddings_model():
    return OpenAIEmbeddings()


def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])
