from typing import Dict, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class ReinforcementLearningAlgorithm(BaseAlgorithm):
    """A reinforcement learning algorithm that solves an MDP formulation of the optimization problem.
    """