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

def extract_code(algorithm_str):
    """Extract code from algorithm string."""
    code = algorithm_str.split("```")[1][6:] # 6 is the length of "python"
    return code

def format_response(
    response: str, 
    variables: list,
    ) -> Dict[str, str]:
        """
        Format string response as dictionary of target variables and values.

        Args:
            response: String response from model
            variables: List of target variables

        Returns:
            Dictionary of target variables and values
        """
        var_dict = {}
        for lines in response.splitlines():
            for var in variables:
                if f'{var}:' in lines:
                    var_dict[var] = lines.split(': ')[1]
        return var_dict

def delete_print_statement(algorithm_str: str):
    """Deletes a print statement from the algorithm."""
    lines = algorithm_str.split('\n')
    print_lines = [i for i, line in enumerate(lines) if 'print' in line]
    if len(print_lines) == 0:
        return algorithm_str
    else:
        line_to_delete = random.choice(print_lines)
        lines.pop(line_to_delete)
        return '\n'.join(lines)

def insert_print_statement(algorithm_str: str, insertion_point: int, print_statement: str):
    """Inserts a print statement at a random location in the algorithm maintaining indentation."""
    lines = algorithm_str.split('\n')
    leading_spaces = len(lines[insertion_point]) - len(lines[insertion_point].lstrip()) # get indentation
    indentation = " " * leading_spaces # create indentation string
    print_statement = f'{indentation}print({print_statement})' 
    lines.insert(insertion_point, print_statement)
    return '\n'.join(lines)