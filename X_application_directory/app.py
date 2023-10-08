import streamlit as st
import openai
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from analyze import AnalyzeGPT, SQL_Query, ChatGPT_Handler
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
import datetime
import sqlite3

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_setting(setting_name, session_name, default_value=""):
    """
    Function to load the setting information from session
    """
    if session_name not in st.session_state:
        st.session_state[session_name] = default_value


load_setting("SQLITE_DB_PATH", "sqlitedbpath", "data/northwind.db")


def saveOpenAI():
    # We can close out the settings now
    st.session_state["show_settings"] = False


def toggleSettings():
    st.session_state["show_settings"] = not st.session_state["show_settings"]


max_response_tokens = None
token_limit = 8000
temperature = 0

st.set_page_config(
    page_title="Natural Language Query", page_icon=":memo:", layout="wide"
)

col1, col2 = st.columns((3, 1))

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    options = ("SQL Query Writing Assistant", "Data Analysis Assistant")
    index = st.radio(
        "Choose the app", range(len(options)), format_func=lambda x: options[x]
    )
    if index == 0:
        system_message = """
        You are an agent designed to interact with a SQL database with schema detail in <<data_sources>>.
        Given an input question, create a syntactically correct {sql_engine} query to run, then look at the results of the query and return the answer.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        Remember to format SQL query as in ```sql\n SQL QUERY HERE ``` in your response.

        """
        few_shot_examples = ""
        extract_patterns = [("sql", r"```sql\n(.*?)```")]
        extractor = ChatGPT_Handler(extract_patterns=extract_patterns)

        faq_dict = {
            "ChatGPT": [
                "Show me revenue by product in ascending order",
                "Show me top 10 most expensive products",
                "Show me net revenue by year. Revenue time is based on shipped date.",
                "For each category, get the list of products sold and the total sales amount",
                "Find Quarterly Orders by Product. First column is Product Name, then year then four other columns, each for a quarter. The amount is order amount after discount",
            ],
        }

    elif index == 1:
        system_message = """
        You are a smart AI assistant to help answer business questions based on analyzing data. 
        You can plan solving the question with one more multiple thought step. At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.
        You are given following utilities to help you retrieve data and commmunicate your result to end user.
        1. execute_sql(sql_query: str): A Python function can query data from the <<data_sources>> given a query which you need to create. The query has to be syntactically correct for {sql_engine} and only use tables and columns under <<data_sources>>. The execute_sql function returns a Python pandas dataframe contain the results of the query.
        2. Use plotly library for data visualization. 
        3. Use observe(label: str, data: any) utility function to observe data under the label for your evaluation. Use observe() function instead of print() as this is executed in streamlit environment. Due to system limitation, you will only see the first 10 rows of the dataset.
        4. To communicate with user, use show() function on data, text and plotly figure. show() is a utility function that can render different types of data to end user. Remember, you don't see data with show(), only user does. You see data with observe()
            - If you want to show  user a plotly visualization, then use ```show(fig)`` 
            - If you want to show user data which is a text or a pandas dataframe or a list, use ```show(data)```
            - Never use print(). User don't see anything with print()
        5. Lastly, don't forget to deal with data quality problem. You should apply data imputation technique to deal with missing data or NAN data.
        6. Always follow the flow of Thought: , Observation:, Action: and Answer: as in template below strictly. 

        """

        few_shot_examples = """
        <<Template>>
        Question: User Question
        Thought 1: Your thought here.
        Action: 
        ```python
        #Import neccessary libraries here
        import numpy as np
        #Query some data 
        sql_query = "SOME SQL QUERY"
        step1_df = execute_sql(sql_query)
        # Replace NAN with 0. Always have this step
        step1_df['Some_Column'] = step1_df['Some_Column'].replace(np.nan,0)
        #observe query result
        observe("some_label", step1_df) #Always use observe() instead of print
        ```
        Observation: 
        step1_df is displayed here
        Thought 2: Your thought here
        Action:  
        ```python
        import plotly.express as px 
        #from step1_df, perform some data analysis action to produce step2_df
        #To see the data for yourself the only way is to use observe()
        observe("some_label", step2_df) #Always use observe() 
        #Decide to show it to user.
        fig=px.line(step2_df)
        #visualize fig object to user.  
        show(fig)
        #you can also directly display tabular or text data to end user.
        show(step2_df)
        ```
        Observation: 
        step2_df is displayed here
        Answer: Your final answer and comment for the question. Also use Python for computation, never compute result youself.
        <</Template>>

        """

        extract_patterns = [
            ("Thought:", r"(Thought \d+):\s*(.*?)(?:\n|$)"),
            ("Action:", r"```python\n(.*?)```"),
            ("Answer:", r"([Aa]nswer:) (.*)"),
        ]
        extractor = ChatGPT_Handler(extract_patterns=extract_patterns)
        faq_dict = {
            "ChatGPT": [
                "Show me daily revenue trends in 2016 per region",
                "Is that true that top 20% customers generate 80% revenue in 2016? What's their percentage of revenue contribution?",
                "Which products have most seasonality in sales quantity in 2016?",
                "Which customers are most likely to churn?",
                "What is the impact of discount on sales? What's optimal discount rate?",
                "Predict monthly revenue for next 6 months starting from June-2018. Do not use Prophet. Show the prediction in a chart together with historical data for comparison.",
            ],
        }

    st.button("Settings", on_click=toggleSettings)

    chat_list = []
    chat_list.append("ChatGPT")
    gpt_engine = st.selectbox("GPT Model", chat_list)
    if gpt_engine == "ChatGPT":
        faq = faq_dict["ChatGPT"]

    option = st.selectbox("FAQs", faq)

    show_code = st.checkbox("Show code", value=False)
    show_prompt = st.checkbox("Show prompt", value=False)
    # step_break = st.checkbox("Break at every step", value=False)
    question = st.text_area("Ask me a question", option)

    if st.button("Submit"):
        sql_query_tool = SQL_Query(db_path=st.session_state.sqlitedbpath)

        if uploaded_file is not None:
            table_name = "legal_questions"
            sql_query_tool.write_csv_to_sql(
                "../src_index/reddit_legal_cluster_test_results.csv", table_name
            )

            analyzer = AnalyzeGPT(
                sql_engine="sqlite",
                content_extractor=extractor,
                sql_query_tool=sql_query_tool,
                system_message=system_message,
                few_shot_examples=few_shot_examples,
                st=st,
                model="gpt-3.5-turbo-16k-0613",
                max_response_tokens=max_response_tokens,
                token_limit=token_limit,
                temperature=temperature,
            )
            if index == 0:
                analyzer.query_run(question, show_code, show_prompt, col1)
            elif index == 1:
                analyzer.run(question, show_code, show_prompt, col1)
            else:
                st.error("Not implemented yet!")
