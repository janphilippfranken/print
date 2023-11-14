
from helpers import extract_code

def insert_debug_prints(initial_solution, language_model):
    """
    Enhances a given solution by inserting print statements for debugging.
    """
    expertise = "You are an expert in debugging Python code, with a strong ability to identify potential issues and insert informative print statements for troubleshooting."

    message = f"""You are given the following Python solution:
```python
{initial_solution}
```
Your task is to insert print statements at strategic points in the code to help identify potential issues or bugs. 
Focus on areas where errors are likely to occur or where the flow of data needs to be tracked. Return the modified code with your print statements included."""
    
    solutions = language_model.batch_prompt(expertise, [message] * language_model.budget)
    modified_solutions = extract_code(solutions)

    return modified_solutions

def generate_print_returns(modified_solutions, utility):
    """
    Simulates the output of print statements in modified solutions.
    """
    print_outputs = []
    for solution in modified_solutions:
        simulated_output = utility.func(solution)[1]
        print_outputs.append(simulated_output)
    
    return print_outputs

def improve_algorithm_with_hints(modified_solutions, hints, utility, language_model):
    """
    Improves a solution according to a utility function and debug hints, and returns the best improvement along with the corresponding hint.
    """
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."

    message_template = """You have been debugging the following solution to a problem:
```python
{modified_solution}
```

Hints from debugging the initial solution: {debug_output}.

You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it."""

    messages = [message_template.format(modified_solution=modified_solution, debug_output=hint) for modified_solution, hint in zip(modified_solutions, hints)]
    solutions = language_model.batch_prompt(expertise, messages)
    solutions = extract_code(solutions)
    best_solution_index, best_solution = max(enumerate(solutions), key=lambda x: utility.func(x[1])[0])

    return messages[best_solution_index], best_solution