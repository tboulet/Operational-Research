from typing import Dict, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class BinaryProgramming(BaseAlgorithm):
    """An algorithm that formalize the BP problem as a LP problem using constraints and then solves it using a LP solver.
    """