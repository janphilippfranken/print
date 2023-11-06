from printlm.language_agents.llm import LLMAgent
from printlm.chat_models.azure import AsyncAzureChatLLM
import os

from parity_task import parity_task
from improve import improver


llm = AsyncAzureChatLLM(azure_endpoint="https://philipp.openai.azure.com/",
                   api_key=os.getenv("OPENAI_API_KEY"),
                   api_version="2023-05-15")

language_model = LLMAgent(
    llm=llm, 
    budget=2, 
    model_id="1", 
    model="gpt-4",
    max_tokens=1000, 
    temperature=0.7, 
    top_p=0.95, 
    n=1)



improve_algorithm, improve_str = improver.get_improver()
parity_task.utility.func, parity_task.utility.str = parity_task.utility.get_utility()
result = improve_algorithm(parity_task.initial_solution, parity_task.utility, language_model)

print(result)