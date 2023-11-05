from pydantic import BaseModel


class TaskPrompt(BaseModel):
    """
    Task Prompt Template
    """
    role: str = "user"
    content: str