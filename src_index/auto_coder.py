import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Optional, Type
from semantic_search import SemanticSearch
from top_n_tool import (
    extract_citation_numbers_in_brackets,
    get_llm_fact_pattern_summary, 
    rerank_with_cross_encoder, 
    create_formatted_input, 
    get_final_answer,
    get_df,
    get_llm
    )
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from llama_index.agent import ReActAgent


# Tool default values
MODEL_NAME = "gpt-3.5-turbo"
TOKEN_LIMIT = 3000


class ResearchSchema(BaseModel):
    query: str = Field(description="the exact text the user wants to query")
    # ton_n: int = Field(description="should be a number")
    # model_name: str = Field(description="should be an OpenAI model name")
    # context_token_limit: int = Field(description="should be a number")


class ResearchPastQuestions(BaseTool):
    name = "research_past_questions"
    description = "useful for taking in a user given query and returning similar cases"
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
        # top_n: int = 5, 
        # model_name: str = MODEL_NAME, 
        # context_token_limit: int = TOKEN_LIMIT,
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
        df = get_df()
        load_dotenv()
        search_engine = SemanticSearch(df)
        llm = get_llm(model="gpt-3.5-turbo")

        # Create instance of SemanticSearch
        search_engine = SemanticSearch(df)

        # Query top n
        top_n_res_df = search_engine.query_similar_documents(
            user_query,
            top_n = 10,
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
            formatted_input = create_formatted_input(rerank_res_df, user_query, context_token_limit=3000)
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
    
    
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.agent.react.types import BaseReasoningStep
from typing import List

    
from typing import Optional
from llama_index.chat_engine.types import AgentChatResponse
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.agent.react.types import BaseReasoningStep
from typing import List, Tuple
from llama_index.llms.base import ChatMessage, MessageRole, ChatResponse


class ReActAgentWrapper(ReActAgent):
    def _process_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        reasoning_steps, is_done = super()._process_actions(output)
        print("Reasoning Steps:", reasoning_steps)
        return reasoning_steps, is_done

    async def _aprocess_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        reasoning_steps, is_done = await super()._aprocess_actions(output)
        print("Reasoning Steps:", reasoning_steps)
        return reasoning_steps, is_done
    # @trace_method("chat")
    def chat(
        self, message: str, use_case: str = "some cool use case", chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=self._memory.get(), current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = self._llm.chat(input_chat, use_case=use_case)  # pass use_case to LLM
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response)
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response
    
    
