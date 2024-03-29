from typing import (
    Any,
    Dict,
    List, 
)

import asyncio
import pickle


from print.chat_models.azure import AsyncAzureChatLLM
from print.language_agents.base import BaseAgent


class LLMAgent(BaseAgent):
    """
    Simple (Chat) LLM Agent class supporting async API calls.
    """
    def __init__(
        self, 
        llm: Any,
        model_id: str, 
        budget: int,
        **model_args,
    ) -> None:
        super().__init__(llm, model_id)
        self.budget = budget
        self.model_args = model_args
        self.all_responses = []
        self.total_inference_cost = 0

    def get_prompt(
        self,
        system_message: str,
        user_message: str,
    ) -> List[Dict[str, str]]:
        """
        Get the (zero shot) prompt for the (chat) model.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return messages
    
    async def get_response(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> Any:
        """
        Get the response from the model.
        """
        self.model_args['temperature'] = temperature
        return await self.llm(messages=messages, **self.model_args)
    
    async def run(
        self, 
        expertise: str,
        message: str,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Runs the Code Agent

        Args:
            expertise (str): The system message to use
            message (str): The user message to use

        Returns:
            A dictionary containing the code model's response and the cost of the performed API call
        """
        # Get the prompt
        messages = self.get_prompt(system_message=expertise, user_message=message)
        # Get the response
        response = await self.get_response(messages=messages, temperature=temperature)
        # Get Cost
        cost = self.calc_cost(response=response)
        print(f"Cost for running {self.model_args['model']}: {cost}")
        # Store response including cost 
        full_response = {
            'response': response,
            'response_str': response.choices[0].message.content,
            'cost': cost
        }
        # Update total cost and store response
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        # Return response_string
        return full_response['response_str']
    
    async def batch_prompt_sync(
        self, 
        expertise: str, 
        messages: List[str],
        temperature: float = 0.7,
    ) -> List[str]:
        """Handles async API calls for batch prompting.

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages

        Returns:
            A list of responses from the code model for each message
        """
        responses = [self.run(expertise, message, temperature) for message in messages]
        return await asyncio.gather(*responses)

    def batch_prompt(
        self, 
        expertise: str, 
        messages: List[str], 
        temperature: float = 0.7,
    ) -> List[str]:
        """=
        Synchronous wrapper for batch_prompt.

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages
            temperature (str): The temperature to use for the API call

        Returns:
            A list of responses from the code model for each message
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(f"Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(expertise, messages, temperature))