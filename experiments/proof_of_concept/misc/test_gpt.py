from print.language_agents.llm import LLMAgent
from print.chat_models.azure import AsyncAzureChatLLM
import os

from experiments.poc.misc.gpt_task_feedback import gpt_task_feedback
from experiments.poc.improver import improver


import os 
from print.language_agents.llm import LLMAgent
from print.chat_models.azure import AsyncAzureChatLLM

llm = AsyncAzureChatLLM(azure_endpoint="https://philipp.openai.azure.com/",
                   api_key=os.getenv("OPENAI_API_KEY"),
                   api_version="2023-05-15")

language_model = LLMAgent(
    llm=llm, 
    budget=1, 
    model_id="1", 
    model="gpt-4",
    max_tokens=2000, 
    temperature=0.7, 
    top_p=0.95, 
    n=1)

improve_algorithm, improve_str = improver.get_improver()
gpt_task_feedback.utility.func, gpt_task_feedback.utility.str = gpt_task_feedback.utility.get_utility()
initial_solution = gpt_task_feedback.get_solution()

from helpers import extract_code




def improve_algorithm(initial_solution, utility, language_model):


    """Improves a solution according to a utility function."""
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."

    debug_message = f"""You are given the following solution to a problem:
```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility.str}
```

The loss of the current solution is {utility.func(initial_solution)[0]} (lower is better).


You must better understand the problems with the current solution by inserting print statements to debug the solution. The output of these print statements will be used to fix the solution using a language model. 
Return the full function including your edits. You can input multiple statements to see where the code breaks and what the variables look like!
The return will be used for downstream improvement."""
    debug_solutions = language_model.batch_prompt(expertise, [debug_message] * language_model.budget)
    debug_solutions = extract_code(debug_solutions)

    
    repair_message_template = """You are given the following solution to a problem:

```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility_str}
```
Hints about the code collected from print statements you may find useful: 

print_statements:
```python
{debug_solution}
```

print_outputs:
{debug_output}.

You must improve the current solution. Use hints and be as creative as you can under the constraints."""
    repair_messages = [repair_message_template.format(
        initial_solution=initial_solution,
        utility_str=utility.str,
        initial_score=utility.func(initial_solution)[0],
        debug_solution=debug_solution,
        debug_output=utility.func(debug_solution)[1],
        ) for debug_solution in debug_solutions]
    
    repair_solutions = language_model.batch_prompt(expertise, repair_messages)
    repair_solutions = extract_code(repair_solutions)
    best_solution_index, best_solution = max(enumerate(repair_solutions), key=lambda x: utility.func(x[1])[0])
    
    # Retrieve the corresponding debug solution
    best_debug_solution = debug_solutions[best_solution_index]

    return best_solution, best_debug_solution



best_sol, best_debug_sol = improve_algorithm(gpt_task_feedback.get_solution(), gpt_task_feedback.utility, language_model)

def improve_algorithm_2(initial_solution, utility, language_model):

    """Improves a solution according to a utility function."""
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."
   
    
    repair_message = f"""You are given the following solution to a problem:

```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility.str}
```

The score of the current solution is {utility.func(initial_solution)[0]}.

You must improve the current solution. Use hints and be as creative as you can under the constraints. Return the full solution including your edits."""
 
    repair_solutions = language_model.batch_prompt(expertise, [repair_message] * language_model.budget)
    print(repair_solutions[0])
    repair_solutions = extract_code(repair_solutions)
    # Find the best solution and its index

    best_solution_index, best_solution = max(enumerate(repair_solutions), key=lambda x: utility.func(x[1])[0])
    # Retrieve the corresponding debug solution
    return best_solution

best_sol2 = improve_algorithm(gpt_task_feedback.get_solution(), gpt_task_feedback.utility, language_model)


import sys

# Reset sys.stdout to its default value, which is the console
sys.stdout = sys.__stdout__

print(best_sol)
print(best_debug_sol)
print(best_sol2)

