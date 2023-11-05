from typing import (
    Any,
    Dict,
    List, 
)

from printlm.agents.base import BaseAgent
from printlm.prompts.code.models import CodePrompt
from printlm.prompts.task.models import TaskPrompt


class CodeAgent(BaseAgent):
    """
    Code agent class.
    """
    def __init__(
        self, 
        llm: Any,
        model_id: str, 
        **model_args,
    ) -> None:
        super().__init__(llm, model_id)
        self.model_args = model_args

    def _get_prompt(
        self,
        code_prompt: CodePrompt,
        task_prompt: TaskPrompt,
    ) -> List[Dict[str, str]]:
        """
        Get the (zero shot) prompt for the (chat) model.
        """
        messages = [
            {"role": code_prompt.role, "content": code_prompt.content},
            {"role": task_prompt.role, "content": task_prompt.content},
        ]
        return messages

    def _get_response(self,
        messages: List[Dict[str, str]],
        ) -> Any:
        """
        Get the response from the model.
        """
        response = self.llm(messages=messages, **self.model_args)
        return response
    
    def run(
        self, 
        code_prompt: CodePrompt,
        task_prompt: TaskPrompt,
    ) -> Dict[str, Any]:
        """Runs the Code Agent

        Args:
            code_prompt (CodePrompt): The code prompt to use
            task_prompt (TaskPrompt): The task prompt to use

        Returns:
            A dictionary containing the code model's response and the cost of the performed API call
        """
        # Get the prompt
        messages = self._get_prompt(code_prompt=code_prompt, task_prompt=task_prompt)
        # Get the response
        response = self._get_response(messages=messages)
        # Get Cost
        cost = self.calc_cost(response=response)

        return {
            'response': response,
            'response_str': response.choices[0].message.content,
            'cost': cost
        }