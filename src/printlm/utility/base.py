from typing import (
    Any,
    Dict,
    List, 
)

from abc import ABC


class BaseUtility(ABC):
    """
    Base agent class for task utility.
    """
    def __init__(
        self, 
        utility_id: str,
    ) -> None:
        """Initializes a utiliy.

        Args:
            utility_id: The unique identifier of the utility
        """
        self.utility_id = utility_id
        self.utility_functions = [] # list of utility algorithms as callable functions
        self.utility_strings = [] # list of utility algorithms as strings
        
    def add_utility(
        self, 
        utility_algorithm, 
        utility_string,
    ) -> None:
        """
        Add an utility to the utility buffer.
        """
        self.utility_functions.append(utility_algorithm)
        self.utility_strings.append(utility_string)

    def get_utility(
        self, 
        i: int = -1,
    ) -> None:
        """
        Get an utility from the utility buffer.
        """
        return self.utility_functions[i], self.utility_strings[i]