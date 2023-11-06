from typing import (
    Any,
    Dict,
    List, 
)

from abc import ABC


class BaseImprover(ABC):
    """
    Base agent class for improvement.
    """
    def __init__(
        self, 
        improver_id: str,
    ) -> None:
        """Initializes an improver model.

        Args:
            improver_id: The unique identifier of the improver
        """
        self.improver_id = improver_id
        self.improve_functions = [] # list of improved algorithms as callable functions
        self.improve_strings = [] # list of improved algorithms as strings
        
    def add_improver(
        self, 
        improve_algorithm, 
        improve_string,
    ) -> None:
        """
        Add an improver to the improver buffer.
        """
        self.improve_functions.append(improve_algorithm)
        self.improve_strings.append(improve_string)

    def get_improver(
        self, 
        i: int = -1,
    ) -> None:
        """
        Get an improver from the improver buffer.
        """
        return self.improve_functions[i], self.improve_strings[i]