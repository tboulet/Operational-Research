from typing import Dict, Type
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.lp_rounding import LinearProgrammingWithAroundSearching
from algorithms.pyomo_solver import PyomoSolver


algo_tag_to_AlgoClass: Dict[str, Type[BaseAlgorithm]] = {
    "Pyomo Solver": PyomoSolver,
    "Linear Programming With Around Searching": LinearProgrammingWithAroundSearching,
}
