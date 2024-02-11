from typing import Dict, Type
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.gplk_solver import GPLK_Solver


algo_tag_to_AlgoClass : Dict[str, Type[BaseAlgorithm]] = {
    "GPLK solution": GPLK_Solver,
}