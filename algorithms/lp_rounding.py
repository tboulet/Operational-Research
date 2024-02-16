import random
from re import L
from typing import Dict, List, Set, Union
import numpy as np

from scipy.optimize import linprog

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class LinearProgrammingWithAroundSearching(BaseAlgorithm):
    """An algorithm that solves the LP relaxation and then rounds the solution to the nearest integer."""

    def initialize_algorithm(self, problem: BaseOptimizationProblem) -> None:
        self.problem = problem
        # Hyperparameters
        self.n_checks_max: int = self.config["n_checks_max"]
        self.stop_if_sol_found: bool = self.config["stop_if_sol_found"]
        self.search_method: str = self.config["search_method"]
        # Internal variables
        self.n_checks_done: int = 0
        self.has_computed_linear_relaxation = False
        self.has_found_feasible_solution = False
        self.checked_all_combinations = False
        self.lp_solution: np.ndarray = None
        self.best_integer_solution: np.ndarray = None
        self.best_value: float = None
        self.combinations: Set[List[int]] = set()

    def run_one_iteration(self) -> Dict[int, Union[int, float]]:

        # Compute the linear relaxation's solution
        if not self.has_computed_linear_relaxation:
            # Get the linear formulation
            (
                variable_names,
                self.c,
                self.A,
                self.b,
                self.A_eq,
                self.b_eq,
                self.variable_types,
                self.variable_bounds,
                self.sense,
            ) = self.problem.get_linear_formulation()
            self.best_value = -np.inf if self.sense == "maximize" else np.inf
            self.bounds = [
                self.variable_bounds[i] for i in range(len(self.variable_bounds))
            ]
            # Solve the linear relaxation
            self.lp_solution, self.value_lp_relaxation, success, status = (
                self.solve_linear_formulation(
                    c=self.c,
                    A=self.A,
                    b=self.b,
                    A_eq=self.A_eq,
                    b_eq=self.b_eq,
                    bounds=self.bounds,
                    sense=self.sense,
                )
            )
            if not success:
                raise ValueError(
                    f"Linear relaxation could not be solved. Status: {status}"
                )
            self.has_computed_linear_relaxation = True
            # Compute the combinations of floor and ceil for each variable
            if self.search_method == "random_without_replacement":
                self.combinations = self.get_floor_or_ceil_combinations(
                    list_float=self.lp_solution,
                    list_variable_types=[
                        self.variable_types[i] for i in range(len(self.variable_types))
                    ],
                    shuffle=True,
                )

        # Method 1 : random search
        if self.search_method == "random":
            integer_solution = self.lp_solution.copy()
            for i in range(len(integer_solution)):
                # If the variable is continuous, we do not need to round it
                if self.variable_types[i] == "continuous":
                    continue
                # If the variable is an int, we do not need to round it
                if int(integer_solution[i]) == integer_solution[i]:
                    continue
                # Else, we round it randomly to the lower or upper integer
                else:
                    integer_solution[i] = random.choice(
                        [np.floor(integer_solution[i]), np.ceil(integer_solution[i])]
                    )

        # Method 2 : random search without replacement
        elif self.search_method == "random_without_replacement":
            if len(self.combinations) == 0:
                self.checked_all_combinations = True
                integer_solution = self.lp_solution
            else:
                integer_solution = self.combinations.pop()

        else:
            raise ValueError(f"Unknown search method {self.search_method}")

        # Check if the solution is feasible and better than the best found so far
        is_feasible, value = self.evaluate_solution_of_linear_formulation(
            integer_solution,
            self.c,
            self.A,
            self.b,
            self.A_eq,
            self.b_eq,
            self.bounds,
        )
        if is_feasible:
            self.has_found_feasible_solution = True
            if (self.sense == "maximize" and value > self.best_value) or (
                self.sense == "minimize" and value < self.best_value
            ):
                self.best_value = value
                self.best_integer_solution = integer_solution

        # Return the best solution found so far (or the LP solution if no feasible solution has been found yet)
        self.n_checks_done += 1
        if self.has_found_feasible_solution:
            return {
                variable_idx: self.best_integer_solution[variable_idx]
                for variable_idx in range(len(integer_solution))
            }
        else:
            return {
                variable_idx: self.lp_solution[variable_idx]
                for variable_idx in range(len(integer_solution))
            }

    def stop_algorithm(self) -> bool:
        if self.checked_all_combinations:
            return True
        if self.stop_if_sol_found and self.has_found_feasible_solution:
            return True
        return self.n_checks_done >= self.n_checks_max

    def get_floor_or_ceil_combinations(
        self,
        list_float: List[float],
        list_variable_types: List[str],
        shuffle=True,
    ) -> List[List[int]]:
        """Get all possible combinations of floor and ceil for a list of floats.

        Args:
            list_float (List[float]): The list of floats to round.
            variable_types (List[str]): The types of the variables.
            shuffle (bool, optional): Whether to shuffle the combinations. Defaults to True.
        """
        if len(list_float) == 0:
            return [[]]
        elif (
            int(list_float[0]) == list_float[0]
            or list_variable_types[0] == "continuous"
        ):
            res = [
                [int(list_float[0])] + comb
                for comb in self.get_floor_or_ceil_combinations(
                    list_float[1:], list_variable_types[1:], shuffle=shuffle
                )
            ]
        else:
            res = [
                [int(list_float[0])] + comb
                for comb in self.get_floor_or_ceil_combinations(
                    list_float[1:], list_variable_types[1:], shuffle=shuffle
                )
            ] + [
                [int(list_float[0]) + 1] + comb
                for comb in self.get_floor_or_ceil_combinations(
                    list_float[1:], list_variable_types[1:], shuffle=shuffle
                )
            ]
        if shuffle:
            random.shuffle(res)
        return res
