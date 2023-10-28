from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain

from helpers import format_response

from helpers import extract_code


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

def choose_print_statement(algorithm_str, utility_str, language_model):
    """Let the language model decide where and what to print for debugging."""
    
    expertise = "Decide where and what to print in the given solution to help with debugging. You can only insert one print statement and only print one variable using the print statement."

    message = f"""Choose a new location and content (i.e., variable to print) for a print statement in the following solution to assist with debugging:

```python
{algorithm_str}
```

You will use the return of the print statement to improve the above solution. Upon improvement, you will be evaluated based on the following utility:

```python
{utility_str}
```

Response format:
Line: <int>
Variable: <variable_name>
""" 
    system_prompt = SystemMessagePromptTemplate.from_template(expertise)
    human_prompt = HumanMessagePromptTemplate.from_template(message)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=language_model, prompt=chat_prompt)
    response = chain.run(stop=['System:'])  # this should return location and content for print statement
    response = format_response(response, variables=['Line', 'Variable'])
    return int(response['Line']), response['Variable']