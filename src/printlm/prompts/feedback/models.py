from pydantic import BaseModel


class FeedbackPrompt(BaseModel):
    """
    Feedback Agent Prompt Template
    """
    id: str = "id of the prompt"
    role: str = "system"
    content: str