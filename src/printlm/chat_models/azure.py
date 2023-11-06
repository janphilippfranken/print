from typing import List, Dict

from openai import AsyncAzureOpenAI

class AzureChatLLM:
    """
    Simple wrapper for an Azure Chat Model.
    """
    def __init__(
        self, 
        api_key: str, 
        azure_endpoint: str, 
        api_version: str, 
        ):
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
        )

    @property
    def llm_type(self):
        return "Azure"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Generates a response from a chat model.
        """
        return await self.client.chat.completions.create(messages=messages, **kwargs)