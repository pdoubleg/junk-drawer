import os
from dotenv import load_dotenv
from llama_index import LLMPredictor, VectorStoreIndex, ServiceContext, StorageContext

from src.utils.gen_utils import get_llm, get_llama_embeddings_model
from src.docstore.read_docs import read_docs

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def docs2vecstore(path, index_id, docname, chunk_chars, overlap, force_pypdf, model_temperature, persist_dir):
    nodes, _ = read_docs(
        path=path, docname=docname, chunk_chars=chunk_chars, overlap=overlap, force_pypdf=force_pypdf,
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
    vector_index.set_index_id(index_id)
    vector_index.storage_context.persist(persist_dir=persist_dir)


if __name__ == "__main__":
    docs2vecstore(
        path="./data/policy_docs/HO3_sample.pdf",
        index_id = "policy_form_collection",
        docname="HO3_Policy",
        chunk_chars=1024,
        overlap=5,
        force_pypdf=False,
        model_temperature=0,
        persist_dir="./_index_storage",
    )
