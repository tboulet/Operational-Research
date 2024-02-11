from typing import Dict, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class LinearProgrammingWithAroundChecking(BaseAlgorithm):
    """An algorithm that solves the LP relaxation and then tries the 2^n_variables integer solutions around the LP solution.
    """