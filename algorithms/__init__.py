from typing import Dict, Type
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.best_fit import BestFitVMPAlgorithm
from algorithms.branch_and_bound import BranchAndBoundAlgorithm
from algorithms.first_fit import FirstFitVMPAlgorithm
from algorithms.first_fit_decreasing import FirstFitDecreasingVMPAlgorithm
from algorithms.lp_search_around import LinearProgrammingWithAroundSearching
from algorithms.pyomo_solver import PyomoSolver


algo_tag_to_AlgoClass: Dict[str, Type[BaseAlgorithm]] = {
    "First-Fit" : FirstFitVMPAlgorithm,
    "First-Fit-Decreasing" : FirstFitDecreasingVMPAlgorithm,
    "Best-Fit" : BestFitVMPAlgorithm,
    "Branch and Bound" : BranchAndBoundAlgorithm,
    "Pyomo Solver": PyomoSolver,
    "Linear Programming With Around Searching": LinearProgrammingWithAroundSearching,
}
