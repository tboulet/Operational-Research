from typing import Dict, Type
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem


problem_tag_to_ProblemClass: Dict[str, Type[BaseOptimizationProblem]] = {
    "VM Placement": VMPlacementProblem,
}
