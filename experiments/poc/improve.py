from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain

from helpers import format_response, extract_code


def improve_algorithm(initial_solution, utility, language_model, n_calls=2):

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
    
    system_prompt = SystemMessagePromptTemplate.from_template(expertise)
    human_prompt = HumanMessagePromptTemplate.from_template(message)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=language_model, prompt=chat_prompt)
    new_solutions = []
    for _ in range(n_calls):
        new_solution = chain.run(stop=['System:'])  
        new_solution = extract_code(new_solution) 
        new_solutions.append(new_solution)
    best_solution = max(new_solutions, key=lambda x: utility.func(x))
    return best_solution