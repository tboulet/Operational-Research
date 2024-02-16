from collections import defaultdict
import itertools
from typing import Dict, List, Tuple, Union
import numpy as np
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem


class VMPlacementProblemEmptyServers(VMPlacementProblem):
    """A variant of the VM placement problem where all servers have the same capacities.
    """

    def __init__(self, config : dict) -> None:
        """Create an instance of the VMPlacementProblemWithIncomp class.
        
        It calls the parent constructor which will intialize the problem with the basic VM placement constraints and
        generate the linear formulation of the problem.
        It then set the server capacities to be the same for all servers (the maximum capacity).
        """
        super().__init__(config)
        ressource_name_to_max_capacity = defaultdict(lambda: 0)
        # Compute the maximum capacity for each ressource
        for i, server_capacity in self.server_index_to_server_capacity.items():
            for ressource_name, capacity in server_capacity.items():
                ressource_name_to_max_capacity[ressource_name] = max(ressource_name_to_max_capacity[ressource_name], capacity)
        # Set the server capacities to be the maximum capacity
        for i in range(self.n_servers):
            for ressource_name, max_capacity in ressource_name_to_max_capacity.items():
                self.server_index_to_server_capacity[i][ressource_name] = max_capacity