from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    Tuple,
)

import random

def extract_code(
        algorithm_strs: List[str],
    ) -> str:
        """Extract code from algorithm string."""
        code = [algorithm_str.split("```")[1][6:] for algorithm_str in algorithm_strs]
        return code

def evaluate_code(
    algorithm_str: str,
    test_cases: List[Tuple[Any, Any]],
) -> List[Any]:
    """Evaluates code against test cases."""
    code = extract_code(algorithm_str)

    return [eval(code)(test_case[0]) == test_case[1] for test_case in test_cases]

def delete_print_statement(algorithm_str: str) -> str:
    """Deletes a print statement from the algorithm."""
    lines = algorithm_str.split('\n')
    print_lines = [i for i, line in enumerate(lines) if 'print' in line]
    if len(print_lines) == 0:
        return algorithm_str
    else:
        line_to_delete = random.choice(print_lines)
        lines.pop(line_to_delete)
        return '\n'.join(lines)

def insert_print_statement(
    algorithm_str: str, 
    insertion_point: int, 
    print_statement: str
) -> str:
    """
    Inserts a print statement at a specific location in the algorithm maintaining indentation.
    """
    lines = algorithm_str.split('\n')
    leading_spaces = len(lines[insertion_point]) - len(lines[insertion_point].lstrip())  # get indentation
    indentation = " " * leading_spaces  # create indentation string
    print_statement = f'{indentation}print({print_statement})' 
    lines.insert(insertion_point, print_statement)
    return '\n'.join(lines)