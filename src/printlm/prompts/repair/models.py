from pydantic import BaseModel


class RepairPrompt(BaseModel):
    """
    Repair Agent Prompt Template
    """
    id: str = "id of the prompt"
    role: str = "system"
    content: str