

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class LP_RoundingAlgorithm(BaseAlgorithm):
    
    def initialize_algorithm(self, problem: BaseOptimizationProblem) -> None:
        return super().initialize_algorithm(problem)