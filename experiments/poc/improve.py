from helpers import extract_code

def improve_algorithm(initial_solution, utility, language_model):
    """Improves a solution according to a utility function."""
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."
   
    message = f"""You are given the following solution to a problem:

```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility.str}
```

You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it."""
    solutions = language_model.batch_prompt(expertise, [message] * language_model.budget)
    solutions = extract_code(solutions)
    _, best_solution = max(enumerate(solutions), key=lambda x: utility.func(x[1])[0])
    return best_solution