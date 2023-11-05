"""
Task Prompt Template
"""
from typing import Dict

from printlm.prompts.task.models import TaskPrompt


TASK_PROMPT: Dict[str, TaskPrompt] = {
    "task_prompt_1": TaskPrompt(
        role="user",
        content="""### QUESTION
{task}""",
    ),
}