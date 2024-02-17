from collections import defaultdict
import itertools
from typing import Dict, List, Tuple, Union
import numpy as np
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem


class VMPlacementProblemEmptyServers(VMPlacementProblem):
    """A variant of the VM placement problem where all servers have the same capacities."""

    def compute_linear_formulation(
        self,
        relax_unique_assignment_constraint: bool,
    ) -> Tuple[
        Dict[int, str],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[int, str],
        Dict[int, Tuple[float, float]],
    ]:
        """Compute the linear formulation of the problem with empty servers.

        For this, the servers are set as having the same capacities, and then the linear formulation is computed as usual.
        """
        ressource_name_to_max_capacity = defaultdict(lambda: 0)
        # Compute the maximum capacity for each ressource
        for i, server_capacity in self.server_index_to_server_capacity.items():
            for ressource_name, capacity in server_capacity.items():
                ressource_name_to_max_capacity[ressource_name] = max(
                    ressource_name_to_max_capacity[ressource_name], capacity
                )
        # Set the server capacities to be the maximum capacity
        for i in range(self.n_servers):
            for ressource_name, max_capacity in ressource_name_to_max_capacity.items():
                self.server_index_to_server_capacity[i][ressource_name] = max_capacity
        return super().compute_linear_formulation(relax_unique_assignment_constraint)
