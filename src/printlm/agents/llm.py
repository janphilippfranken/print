from typing import (
    Any,
    Dict,
    List, 
)

import asyncio

import pickle

from printlm.agents.base import BaseAgent


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

    def _get_prompt(
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
    
    async def _get_response(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> Any:
        """
        Get the response from the model.
        """
        self.model_args['temperature'] = temperature
        return await self.llm(messages=messages, **self.model_args)
    
    def _store_response(
        self, 
        response: Any,
    ) -> None:
        """
        Writes the full response object to a file.
        """
        try:
            with open(f"{self.model_id}.pkl", "wb") as f:
                pickle.dump(response, f)
        except Exception as e:
            print(f"Error occurred while saving the response: {e}")
    
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
        messages = self._get_prompt(system_message=expertise, user_message=message)
        # Get the response
        response = await self._get_response(messages=messages, temperature=temperature)
        # Get Cost
        cost = self.calc_cost(response=response)
        print(f"Cost: {cost}")
        # Store response including cost 
        full_response = {
            'response': response,
            'response_str': response.choices[0].message.content,
            'cost': cost
        }
        self._store_response(response=full_response)
        # Return str response
        return response.choices[0].message.content
    
    async def batch_prompt(
        self, 
        expertise: str, 
        messages: List[str],
        temperature: float = 0.7,
    ) -> List[str]:
        """Allows for async API calls

        Args:
            expertise (str): The system message to use
            messages (List[str]): A list of user messages

        Returns:
            A list of responses from the code model for each message
        """
        responses = [self.run(expertise, message, temperature) for message in messages]
        return await asyncio.gather(*responses)