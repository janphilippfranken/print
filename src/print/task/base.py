from typing import Callable, List

from abc import ABC

from print.utility.base import BaseUtility


class BaseTask(ABC):

    def __init__(
        self,
        task_id: str, 
        initial_solution: str,
        utility: BaseUtility,
    ) -> None:
        """
        Initializes a task.
        """
        self.task_id: str = task_id
        self.initial_solution: str = initial_solution
        self.utility: BaseUtility = utility
        self.solutions = []
    
    def add_solution(
        self, 
        solution: str,
    ) -> None:
        self.solutions.append(solution)

    def get_solution(
        self, 
        i: int = -1,
    ) -> str:
        return self.solutions[i] if self.solutions else self.initial_solution