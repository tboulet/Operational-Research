from typing import Dict, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class LinearProgrammingWithRounding(BaseAlgorithm):
    """An algorithm that solves the LP relaxation and then rounds the solution to the nearest integer.
    """