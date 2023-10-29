from typing import (
    Any,
    Dict,
    List, 
)

from abc import ABC, abstractmethod

from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models.base import BaseChatModel

from scai.memory.buffer import ConversationBuffer
from scai.memory.memory import ChatMemory

class BaseAgent(ABC):
    """
    Base agent class.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
    ) -> None:
        """Initializes a chat model.

        Args:
            llm: BaseChatModel
        """
        self.llm = llm
        
    @abstractmethod
    def _get_prompt(self) -> ChatPromptTemplate:
        """
        Get the prompt fed into the model. 
        """
        raise NotImplementedError

    @abstractmethod
    def _get_response(self) -> str:
        """
        Get the response from the model. 
        """
        raise NotImplementedError   

    @abstractmethod 
    def run(self) -> Dict[str, Any]:
        """
        Run the model.
        """
        raise NotImplementedError