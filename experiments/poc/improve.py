from helpers import extract_code

from printlm.improvers.base import BaseImprover

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

improve_str = """def improve_algorithm(initial_solution, utility, language_model):

    "Improves a solution according to a utility function."
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."

    message = f\"\"\"Improve the following solution:
```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility.str}
```
You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it\"\"\"
    new_solutions = language_model.batch_prompt(expertise, [message] * language_model.budget)
    new_solutions = extract_code(new_solutions)
    best_solution = max(new_solutions, key=lambda x: utility.func(x))
    return best_solution"""

improver = BaseImprover(improver_id="1")
improver.add_improver(improve_algorithm=improve_algorithm, improve_string=improve_str)