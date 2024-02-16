from typing import Dict, Type
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem
from problems.vmp_splittable import VMPlacementProblemSplittable
from problems.vmp_with_incompatibilities import VMPlacementProblemWithIncomp
from problems.vmp_empty_servers import VMPlacementProblemEmptyServers

problem_tag_to_ProblemClass: Dict[str, Type[BaseOptimizationProblem]] = {
    "VM Placement": VMPlacementProblem,
    "VM Placement with Incompatibilities": VMPlacementProblemWithIncomp,
    "VM Placement Empty Servers": VMPlacementProblemEmptyServers,
    "VM Placement Splittable" : VMPlacementProblemSplittable,
}
