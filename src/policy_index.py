import os
from dotenv import load_dotenv
from llama_index import LLMPredictor, VectorStoreIndex, ServiceContext, StorageContext
from utils.gen_utils import get_llm, get_llama_embeddings_model
from docstore.read_docs import read_docs

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def main(path, docname, chunk_chars, overlap, model_temperature, persist_dir):
    nodes, _ = read_docs(
        path=path, docname=docname, chunk_chars=chunk_chars, overlap=overlap
    )

    llm = get_llm(model_temperature=model_temperature)
    llm_predictor_chat = LLMPredictor(llm=llm)
    embed_model = get_llama_embeddings_model()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor_chat, embed_model=embed_model, chunk_size=1024
    )

    storage_context = StorageContext.from_defaults()

    vector_index = VectorStoreIndex(
        nodes,
        service_context=service_context,
        storage_context=storage_context,
    )
    vector_index.set_index_id(docname)
    vector_index.storage_context.persist(persist_dir=persist_dir)


if __name__ == "__main__":
    main(
        path="./data/policy_docs/HO3_sample.pdf",
        docname="HO3_Policy",
        chunk_chars=1024,
        overlap=5,
        model_temperature=0,
        persist_dir="./ho3_index_storage",
    )
