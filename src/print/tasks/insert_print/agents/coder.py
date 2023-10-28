from typing import (
    Any,
    Dict,
    List, 
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel

from print.tasks.insert_print.agents.base import BaseAgent

from print.tasks.insert_print.prompts.coder.models import CoderPrompt
from print.tasks.insert_print.prompts.task.models import TaskPrompt


class CoderAgent(BaseAgent):
    """
    LLM Chain for running the Coder.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
    ) -> None:
        super().__init__(llm)

    def _get_prompt(
        self,
        coder_prompt: CoderPrompt,
        task_prompt: TaskPrompt,
    ) -> ChatPromptTemplate:
        """
        Returns the prompt template for the coder

        Args:
            coder_prompt: (CoderPrompt) The coder prompt.
            task_prompt: (TaskPrompt) The task prompt.

        Returns:
            ChatPromptTemplate
        """
        system_prompt_template = SystemMessagePromptTemplate.from_template(f"{task_prompt.task}\n\n{coder_prompt.scaffold}")
        return ChatPromptTemplate.from_messages([system_prompt_template])
       
    def _get_response(
        self,
        chat_prompt_template: ChatPromptTemplate,
        system_message: str,
        task_prompt: TaskPrompt,
    ) -> str:
        """
        Returns the response from the coder.

        Args:
            chat_prompt_template: (ChatPromptTemplate) The chat prompt template.
            system_message: (str) The system message.
            task_prompt: (TaskPrompt) The task prompt.

        Returns:
            str
        """
        chain = LLMChain(llm=self.llm, prompt=chat_prompt_template)
        response = chain.run(strategy=system_message,
                             task=task_prompt.buyer_task,
                             stop=['System:'])   
        response = self._format_response(response, ['Choice', 'Reason'])
        return response
        
    def run(
        self, 
        buffer: ConversationBuffer, 
        buyer_prompt: BuyerPrompt, 
        task_prompt: TaskPrompt, 
        turn: int,
        distance_apple: float,
        distance_orange: float,
        reward_apple: float,
        reward_orange: float,
        buyer_level: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Runs the buyer

        Args:
            buffer (ConversationBuffer): The conversation buffer.
            buyer_prompt (buyerPrompt): The buyer prompt.
            task_prompt (TaskPrompt): The task prompt.
            distance_apple (float): The distance of the apple.
            distance_orange (float): The distance of the orange.
            reward_apple (float): The reward of the apple.
            reward_orange (float): The reward of the orange.
            turn (int): The turn number.
            test_run (bool, optional): Whether to run a test run. Defaults to False.
            verbose (bool, optional): Whether to print the buyer's response. Defaults to False.

        Returns:
            A dictionary containing the buyer's response, input prompt, and all other metrics we want to track.
        """
        system_message = self._get_chat_history(buffer, memory_type="system")['system'][-1]['response'][f'system_message_buyer_{buyer_level}'] # the last system message in the chat history (i.e. instructions)
        chat_prompt_template =  self._get_prompt(buffer, buyer_prompt, task_prompt)
        prompt_string = chat_prompt_template.format(strategy=system_message,
                                                    task=task_prompt.buyer_task,
                                                    distance_apple=distance_apple,
                                                    distance_orange=distance_orange,
                                                    reward_apple=reward_apple,
                                                    reward_orange=reward_orange)
      
        response = self._get_response(chat_prompt_template, system_message, task_prompt, distance_apple=distance_apple, distance_orange=distance_orange, reward_apple=reward_apple, reward_orange=reward_orange)
        
        if verbose:
            print('===================================')
            print(f'buyer {str(self.model_id)} turn {turn}')
            print(prompt_string)
            print(response)


        return {
            'prompt': prompt_string, 
            'response': response, 
            'turn': turn
        }
    