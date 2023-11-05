"""
Repair Agent Prompt Template
"""
from typing import Dict

from printlm.prompts.repair.models import RepairPrompt


REPAIR_PROMPT: Dict[str, RepairPrompt] = {
    "repair_prompt_1": RepairPrompt(
        id="repair_prompt_1",
        role="system",
        content="""You are a helpful programming assistant and an expert Python programmer. You are helping a user write a
program to solve a problem. The user has written some code, but it has some errors and is not passing
the tests. The user has spent some time debugging the program and will provide you with a concise
textual explanation of what is wrong with the code. You will use this explanation to generate a fixed
version of the program. Put your fixed program within code delimiters, for example: ```python
# YOUR CODE HERE
```.""" ,
    ),
}