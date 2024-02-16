from typing import Dict, Type
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem
from problems.vmp_with_incompatibilities import VMPlacementProblemWithIncomp


problem_tag_to_ProblemClass: Dict[str, Type[BaseOptimizationProblem]] = {
    "VM Placement": VMPlacementProblem,
    "VM Placement with Incompatibilities": VMPlacementProblemWithIncomp,
}
