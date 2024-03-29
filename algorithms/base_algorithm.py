from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
from scipy.optimize import linprog

from problems.base_problem import BaseOptimizationProblem


class BaseAlgorithm(ABC):
    """Base class for any optimization algorithm. It is an abstract class that defines the interface for any optimization algorithm."""

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def initialize_algorithm(self, problem: BaseOptimizationProblem) -> None:
        """Initialize the algorithm with the given problem.

        Args:
            problem (BaseOptimizationProblem): The problem to solve.
        """

    @abstractmethod
    def run_one_iteration(self) -> Tuple[
        bool,
        Dict[int, Union[int, float]],
    ]:
        """Run one iteration of the algorithm (if the algorithm can be decomposed into iterations).
        At the end of the iteration, the algorithm should be able to propose an (possibly suboptimal or even unvalid) solution to the VMP problem.

        If the algorithm has found no valid solution, it should return (False, Anything), for example (False, None).

        The running of each algorithm is atomized in iteration so that its possible to track the evolution of the solution performance over time if needed.

        Returns:
            bool: A boolean that indicates if the solution is feasible according to the algorithm.
            Dict[int, Union[int, float]]: A dictionary that maps each variable index to its value.
        """

    @abstractmethod
    def stop_algorithm(self) -> bool:
        """Return True if the algorithm should stop, False otherwise."""

    # Helper methods

    def solve_linear_formulation(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        bounds: List[Tuple[Optional[float], Optional[float]]],
        sense: str,
        solver_name: str = "scipy",
    ) -> Tuple[np.ndarray, float, bool, int]:
        """Solve a linear optimization problem.

        Args:
            c (np.ndarray): the cost vector : J = c^T * x
            A (np.ndarray): the inequality matrix : A * x <= b
            b (np.ndarray): the inequality vector : A * x <= b
            A_eq (np.ndarray): the equality matrix : A_eq * x = b_eq
            b_eq (np.ndarray): the equality vector
            bounds (List[Tuple[Optional[float], Optional[float]]]): a list of tuples that contains the lower and upper bounds of each variable
            sense (str): either "minimize" or "maximize"
            solver_name (str, optional): The solver method. Defaults to "scipy".

        Returns:
            np.ndarray: the solution of the optimization problem, as a (n_variables,) array
            float: the value of the objective function
            bool: True if the optimization was successful
            int: the status of the optimization, following scipy's linprog status codes
        """
        if solver_name == "scipy":
            assert sense in ["minimize", "maximize"], f"Unknown sense {sense}"
            c_modified = c if sense == "minimize" else -c
            res = linprog(
                c=c_modified,
                A_ub=A if A.size > 0 else None,
                b_ub=b if b.size > 0 else None,
                A_eq=A_eq if A_eq.size > 0 else None,
                b_eq=b_eq if b_eq.size > 0 else None,
                bounds=bounds,
                method="highs",
            )
            return res.x, res.fun, res.success, res.status

        else:
            raise ValueError(f"Unknown solver {solver_name}")

    def evaluate_solution_of_linear_formulation(
        self,
        x_solution: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        bounds: List[Tuple[Optional[float], Optional[float]]],
    ) -> Tuple[bool, float]:
        """Evaluate the solution of the optimization problem.

        Args:
            x_solution (np.ndarray): the solution of the optimization problem, as a (n_variables,) array
            c (np.ndarray): the cost vector : J = c^T * x
            A (np.ndarray): the inequality matrix : A * x <= b
            b (np.ndarray): the inequality vector : A * x <= b
            A_eq (np.ndarray): the equality matrix : A_eq * x = b_eq
            b_eq (np.ndarray): the equality vector
            bounds (List[Tuple[Optional[float], Optional[float]]]): a list of tuples that contains the lower and upper bounds of each variable

        Returns:
            Tuple[bool, float]: a tuple that contains a boolean that indicates if the solution is feasible and the value of the objective function
        """
        value = np.dot(c, x_solution)
        # Check if the solution is in the bounds
        for i, (lower_bound, upper_bound) in enumerate(bounds):
            if lower_bound is not None and x_solution[i] < lower_bound:
                return False, value
            if upper_bound is not None and x_solution[i] > upper_bound:
                return False, value
        # Check if the solution satisfies the inequalities
        if A.size > 0:
            if np.any(np.dot(A, x_solution) > b):
                return False, value
        # Check if the solution satisfies the equalities
        if A_eq.size > 0:
            if np.any(np.dot(A_eq, x_solution) != b_eq):
                return False, value
        # If all the checks are passed, the solution is feasible
        return True, value
