from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np


class BaseOptimizationProblem(ABC):
    """Base class for any optimization problem."""

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def get_linear_formulation(
        self,
    ) -> Tuple[
        List[str],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str],
        List[Tuple[float, float]],
        str,
    ]:
        """Generate the general linear formulation of the problem.

        Returns:
            List[str] : A dictionary that maps each variable index to its name.
            np.ndarray : The cost vector c.
            np.ndarray : The inequality matrix A.
            np.ndarray : The inequality vector b.
            np.ndarray : The equality matrix A_eq.
            np.ndarray : The equality vector b_eq.
            List[str] : A dictionary that maps each variable index to its type (either "integer" or "continuous").
            List[Tuple[float, float]] : A dictionary that maps each variable index to its lower and upper bounds.
            str : the sense of the optimization problem ("maximize" or "minimize").
        """

    @abstractmethod
    def apply_solution(self, solution: Dict[int, Union[int, float]]) -> Union[bool, float]:
        """Apply the solution to the problem and give feedback on the solution.

        You can give as input a dict that only contains the variables that represents the decision variables of the
        problem (and not the intermediary variables). If the problem can construct the decision from those variables,
        this method will give feedback on the decision, however the full feasability of the solution will not be
        guaranteed, as the intermediary variables are not checked.

        Args:
            solution (Dict[int, Union[int, float]]) : The solution to apply to the problem, as a mapping from variable index to value.

        Returns:
            bool : True if the solution is valid, False otherwise.
            float : The cost of the solution, or np.inf if the solution is not valid. If the given solution
            is partial, the cost returned is no guarantee to be the cost of the full solution.
        """
        # Usefull code regarding non-full solutions
        n = solution.shape[0]
        if n < self.n_variables:
            print(
                "WARNING: The given solution does not have the right number of variables, no guarantee of validity."
            )
            if not (self.c[n:] == 0).all():
                print(
                    "WARNING: The returned objective value does not correspond to the given solution, as variables were not taken into account and cost matrix is non-null for those variables."
                )
