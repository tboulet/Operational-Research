from typing import Dict, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class MCTS(BaseAlgorithm):
    """A MCTS algorithm that solves a graph tree formulation of the optimization problem.
    """