"""
Code Agent Prompt Template
"""
from typing import Dict

from printlm.prompts.code.models import CodePrompt


CODE_PROMPTS: Dict[str, CodePrompt] = {
    "code_prompt_1": CodePrompt(
        id="code_prompt_1",
        role="system",
        content="""You are an expert computer science researcher and programer. You will be given a score function and have to come up with a solution which is evaluated by the score function.
        Return your solution using the following format ```python <Your Solution>```. Do not call your solution, just define it. Call the program "algorithm.""" ,
    ),
}