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