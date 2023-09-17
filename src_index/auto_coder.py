from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.agent.react.types import BaseReasoningStep
from typing import List

    
from typing import Optional
from llama_index.agent import ReActAgent
from llama_index.chat_engine.types import AgentChatResponse
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.agent.react.types import BaseReasoningStep
from typing import List, Tuple
from llama_index.llms.base import ChatMessage, MessageRole, ChatResponse

import streamlit as st



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

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ):
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
            chat_response = self._llm.chat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response)
            current_reasoning.extend(reasoning_steps)
            yield reasoning_steps  # yield the reasoning steps
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        yield response  # yield the final response