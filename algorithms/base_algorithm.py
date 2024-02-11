from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np

from problems.base_problem import BaseOptimizationProblem


class BaseAlgorithm(ABC):
    """Base class for any optimization algorithm. It is an abstract class that defines the interface for any optimization algorithm."""

    def __init__(self, config_algo: Dict):
        self.config_algo = config_algo

    @abstractmethod
    def initialize_algorithm(self, problem: BaseOptimizationProblem) -> None:
        """Initialize the algorithm with the given problem.

        Args:
            problem (BaseOptimizationProblem): The problem to solve.
        """

    @abstractmethod
    def run_one_iteration(self) -> Dict[int, Union[int, float]]:
        """Run one iteration of the algorithm (if the algorithm can be decomposed into iterations).
        At the end of the iteration, the algorithm should be able to propose an (possibly suboptimal or even unvalid) solution to the VMP problem.
        
        The running of each algorithm is atomized in iteration so that its possible to track the evolution of the solution performance over time.
        
        Returns:
            Dict[int, Union[int, float]]: A dictionary that maps each variable index to its value.
        """
        
    @abstractmethod
    def stop_algorithm(self) -> bool:
        """Return True if the algorithm should stop, False otherwise."""