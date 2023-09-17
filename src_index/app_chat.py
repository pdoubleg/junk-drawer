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
from pathlib import Path
from typing import List, Tuple, Optional
from io import StringIO
import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".st_langchain.db")

from llama_index.agent import ReActAgent
from llama_index.tools.function_tool import FunctionTool

from llama_index.llms import LangChainLLM
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Tool, initialize_agent
from langchain import LLMMathChain
from langchain.callbacks import StreamlitCallbackHandler

from html_templates import css, user_template, bot_template

from custom_tools import ResearchPastQuestions

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


def display_description():
    """Displays the description of the app."""
    st.markdown("<h4 style='text-align: left;'>Work with an AI research assistant</h4>", unsafe_allow_html=True)
    st.write(
        """
        Why use an AI agent?
        - 🧰 Agents have access to tools 
        - 🔧 Tools are small programs that do specific tasks
        - 👉 You give the instructions, and the agent figures out which tool, or set of tools to use
        """
    )
    
    
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
        "<h2 style='text-align: left;'>GPT Research Agent 📚</h2>",
        unsafe_allow_html=True,
    )
    
    # Add a sidebar dropdown for model selection
    model = st.sidebar.selectbox(
        'Select Model',
        ('gpt-3.5-turbo', 'gpt-4')
    )

    load_dotenv()

    llm = ChatOpenAI(model=model, temperature=0.0)
    memory = ConversationBufferMemory(memory_key="chat_history")
    df = get_df()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    research_past_questions = ResearchPastQuestions(df=df)
    
    tools = [
        Tool(
            name ='Calculator and Math Tool',
            func=llm_math_chain.run,
            description='Useful for mathematical questions and operations'
        ),
        Tool(
            name ='Legal Questions Research Tool',
            func=research_past_questions.run,
            description='Useful for researching legal questions',
            return_direct = False
        )
    ]
    
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True, 
                             memory=memory,
                             )
    
    
    # Add a reset button to the sidebar
    reset_button = st.sidebar.button("Reset & Clear")
    if reset_button:
        st.session_state.clear()  # Clear the session state to reset the app
        agent = None
    
    display_description()
    b_get_feedback()
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
    
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = []
    #         {"role": "assistant", "content": "Ask me a question"}
    # ]
    
    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
 
    if query:= st.chat_input(placeholder="Ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": query})
        st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
        response = agent.run(query, callbacks=[st_callback])
        # Append the response to the session_state messages
        # st.session_state.messages.append({"role": "assistant", "content": st_callback})
        st.write(st_callback)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display conversation in reverse order
        for message in reversed(st.session_state.messages):
            if message["role"] == "user": 
                st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else: 
                st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)


if __name__ == "__main__":
    app()