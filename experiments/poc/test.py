from printlm.agents.llm import LLMAgent
from printlm.chat_models.azure import AzureChatLLM
import os

from parity_task import task_utility, algorithm_str
from improve import improver
from helpers import extract_code

llm = AzureChatLLM(azure_endpoint="https://philipp.openai.azure.com/",
                   api_key=os.getenv("OPENAI_API_KEY"),
                   api_version="2023-05-15")


language_model = LLMAgent(llm=llm, budget=2, model_id="1", model="gpt-4", max_tokens=1000, temperature=0.7, top_p=0.95, n=1)

def improve_algorithm(initial_solution, utility, language_model):

    """Improves a solution according to a utility function."""
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."

    message = f"""Improve the following solution:
```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility.str}
```
You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it."""
    new_solutions = language_model.batch_prompt(expertise, [message] * language_model.budget)
    new_solutions = extract_code(new_solutions)
    best_solution = max(new_solutions, key=lambda x: utility.func(x))
    return best_solution

result = improve_algorithm(algorithm_str, task_utility, language_model)

print(result)