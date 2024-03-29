from helpers import extract_code

import numpy as np

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
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it. 
You algorithm has to run within max of 2 seconds and you are not allwed to use external libraries besides numpy."""
    try:
        solutions = language_model.batch_prompt(expertise, [message] * language_model.budget)
    except:
        return "", 0
    solutions = extract_code(solutions)
    best_solution = max(solutions, key=lambda solution: utility.func(solution)[0])
    average_utility = np.mean([utility.func(solution)[0] for solution in solutions]) 
    print([utility.func(solution)[0] for solution in solutions])
    return best_solution, average_utility