from typing import List, Dict

import os
import openai

class AzureChatLLM:
    """
    Simple wrapper for an Azure Chat Model.
    """
    def __init__(
        self, 
        api_key: str, 
        api_base: str, 
        api_version: str, 
        api_type: str,
        ):
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.api_type = api_type

        # openai api
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_type = self.api_type

    @property
    def llm_type(self):
        return "Azure"

    def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Generates a response from a chat model.
        """
        response = openai.ChatCompletion.create(
            messages=messages, 
            **kwargs)
        
        return response