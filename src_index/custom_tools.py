import os
import pandas as pd
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Optional, Type
from semantic_search import SemanticSearch
from top_n_tool import (
    extract_citation_numbers_in_brackets,
    get_llm_fact_pattern_summary, 
    rerank_with_cross_encoder, 
    create_formatted_input, 
    add_month_year_to_df,
    get_final_answer,
    get_llm
    )
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun


# Tool default values
MODEL_NAME = "gpt-3.5-turbo"
TOKEN_LIMIT = 3000
DATA_PATH = "reddit_legal_cluster_test_results.parquet"


class ResearchSchema(BaseModel):
    query: str = Field(description="the exact text the user wants to query")
    ton_n: Optional[int] = Field(description="should be a number")
    model_name: Optional[str] = Field(description="should be an OpenAI model name")
    context_token_limit: Optional[int] = Field(description="should be a number")


class ResearchPastQuestions(BaseTool):
    df: pd.DataFrame = Field(...)

    @validator('df')
    def must_be_a_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('df must be a pandas DataFrame')
        return v

    def __init__(self, **data):
        super().__init__(**data)
    name = "Research Past Questions"
    description = "useful for finding top n most similar text for a specified query. if given a query it should be passed to this tool unedited."
    return_direct = True
    args_schema: Type[ResearchSchema] = ResearchSchema
    handle_tool_error = True

  
    @staticmethod
    def _handle_error(error: ToolException) -> str:
        return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool."
        )

    def _run(
        self, 
        user_query: str, 
        top_n: int = 5, 
        model_name: str = MODEL_NAME, 
        context_token_limit: int = TOKEN_LIMIT,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Function to run the entire process end-to-end.
        
        Args:
            user_query (str): The user's text query.
            model_name (str): The name of the model to use.
            token_limit (int): The maximum number of tokens for the context.
        Returns:
            str: A string containing the user query, model response, and cited sources.
        """
        # Read in a df
        # df = pd.read_parquet("reddit_legal_cluster_test_results.parquet")
        load_dotenv()
        llm = get_llm(model=model_name)

        # Create instance of SemanticSearch
        search_engine = SemanticSearch(self.df)

        # Query top n
        top_n_res_df = search_engine.query_similar_documents(
            user_query,
            top_n = top_n,
            filter_criteria = None,
            use_cosine_similarity = True,
            similarity_threshold = 0.93)

        # Run get_llm_fact_pattern_summary
        try:
            top_n_res_df = get_llm_fact_pattern_summary(df=top_n_res_df, text_col_name="body")
        except Exception as e:
            raise ToolException(f"Error in get_llm_fact_pattern_summary: {e}")
            return

        # Run rerank_with_cross_encoder
        try:
            rerank_res_df = rerank_with_cross_encoder(top_n_res_df, user_query, 'summary')
        except Exception as e:
            raise ToolException(f"Error in rerank_with_cross_encoder: {e}")
            return

        # Run create_formatted_input
        try:
            formatted_input = create_formatted_input(rerank_res_df, user_query, context_token_limit=context_token_limit)
        except Exception as e:
            raise ToolException(f"Error in create_formatted_input: {e}")
            return

        # Run get_final_answer
        try:
            response = get_final_answer(formatted_input, llm)
        except Exception as e:
            raise ToolException(f"Error in get_final_answer: {e}")
            return
        
        
        # Create a string containing the user query, model response, and cited sources
        result = f"## New Query:\n{user_query}\n## Model Response:\n{response}\n"
        citation_numbers = extract_citation_numbers_in_brackets(response)
        for citation in citation_numbers:
            i = int(citation) - 1  # convert string to int and adjust for 0-indexing
            title = rerank_res_df.iloc[i]["llm_title"]
            link = f"{rerank_res_df.iloc[i]['full_link']}"
            venue = rerank_res_df.iloc[i]["State"]
            date = rerank_res_df.iloc[i]["datestamp"]
            number = rerank_res_df.iloc[i]["index"]
            result += f"##### {[i+1]} [{title}]({link}) - {venue}, {date}, Number: {number}\n"

        return result
    
    
    async def _arun(
        self, 
        user_query: str, 
        top_n: int = 5, 
        model_name: str = MODEL_NAME, 
        context_token_limit: int = TOKEN_LIMIT,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("research_past_questions does not support async")
    
