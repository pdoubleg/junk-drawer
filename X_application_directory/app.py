__version__ = "0.0.01"
app_name = "LibertyGPT Sandbox"

import streamlit as st
st.set_page_config(page_title="LibertyGPT Sandbox", layout='wide')
st.title('LibertyGPT Sandbox')
from llama_index import LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.langchain_helpers.agents import create_llama_chat_agent, create_llama_agent 
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine


ho3_directory = "../_policy_index_metadatas"
doi_directory = "../_index_storage"
uniform_building_codes = "../_property_index_storage"


st.write("This sandbox is powered by :statue_of_liberty:**LibertyGPT**, 🦜[LangChain](https://langchain-langchain.vercel.app/docs/get_started/introduction.html) and :llama:[Llama-Index](https://gpt-index.readthedocs.io/en/latest/index.html)", 
          unsafe_allow_html=True)


def get_llm(temperature=0):
    return ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")


def get_llm_predictor(temperature=0):
    return LLMPredictor(ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo"))


def initialize_index(storage_directory):
    llm = get_llm_predictor()

    service_context = ServiceContext.from_defaults(llm_predictor=llm)

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=storage_directory),
        service_context=service_context,
    )
    return index

ho3_index = initialize_index(storage_directory=ho3_directory)
doi_index = initialize_index(storage_directory=doi_directory)
bldg_code_index = initialize_index(storage_directory=uniform_building_codes)



ho3_agent_tool = [
    Tool(
        name="ho3_query_engine",
        func=lambda q: str(ho3_index.as_query_engine(
            similarity_top_k=5,
            streaming=True).query(q)),
        description="useful for when you want to answer questions about homeowner's insurance coverage.",
        return_direct=False,
    ),
    Tool(
        name="doi_query_engine",
        func=lambda q: str(doi_index.as_query_engine(
            similarity_top_k=5,
            streaming=True).query(q)),
        description="useful for when you want to answer questions about insurance regulation such as rules, regulations, or statutes.",
        return_direct=False,
    ),
        Tool(
        name="bldg_codes_query_engine",
        func=lambda q: str(bldg_code_index.as_query_engine(
            similarity_top_k=5,
            streaming=True).query(q)),
        description="useful for when you want to answer technical questions about building, consruction, and renovation requirements.",
        return_direct=False,
    ),
]


llm = OpenAI(temperature=0, streaming=True)

agent = initialize_agent(
    ho3_agent_tool, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)



hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
