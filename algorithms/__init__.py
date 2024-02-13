from typing import Dict, Type
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.pyomo_solver import PyomoSolver


algo_tag_to_AlgoClass : Dict[str, Type[BaseAlgorithm]] = {
    "Pyomo Solver": PyomoSolver,
}