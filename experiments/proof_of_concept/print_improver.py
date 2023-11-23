
from helpers import extract_code

def insert_prints(initial_solution, language_model):
    """
    Enhances a given solution by inserting print statements for debugging.
    """
    expertise = "You are an expert in debugging Python code, with a strong ability to identify potential issues and insert informative print statements for troubleshooting."

    message = f"""You are given the following Python solution:
```python
{initial_solution}
```
Your task is to insert print statements at strategic points in the code to help identify potential issues or bugs. 
Focus on areas where errors are likely to occur or where the flow of data needs to be tracked. Return the modified code with your print statements included. Important: While you should include multiple print statments, you should be careful to not print more than eg. 20 lines of output as there is a constraint to use no more than 1000 characters of feedback overall!"""
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

def print_improve_algorithm(initial_solution, print_statements, debug_outputs, utility, language_model):
    """Improves a solution according to a utility function."""
    
    expertise = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."
   
    message = """You are given the following solution to a problem:

```python
{initial_solution}
```

You will be evaluated based on this score function:
```python
{utility_str}
```

You should also consider the following print statements I have inserted:

{print_statement}

As well as their output:
{debug_output}


Considering the initial solution and debugging information, you must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it. 
You algorithm has to run within max of 2 seconds and you are not allwed to use external libraries besides numpy."""
    messages = [message.format(initial_solution=initial_solution, utility_str=utility.str, print_statement=print_statement, debug_output=debug_output) for print_statement, debug_output in zip(print_statements, debug_outputs)]
    try:
        solutions = language_model.batch_prompt(expertise, messages)
    except:
        return "None", 0, 0
    solutions = extract_code(solutions)
    best_solution_index, best_solution = max(enumerate(solutions), key=lambda x: utility.func(x[1])[0])
    average_utility = sum([utility.func(solution)[0] for solution in solutions]) / len(solutions)
    return best_solution, best_solution_index, average_utility