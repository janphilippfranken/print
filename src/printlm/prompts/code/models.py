from pydantic import BaseModel


class CodePrompt(BaseModel):
    """
    Code Agent Prompt Template
    """
    id: str = "id of the prompt"
    role: str = "system"
    content: str