from typing import Dict, Tuple, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem
from src.sort_servers import (
    get_name_ressource_to_max_capacity,
    get_server_averaged_normalized_capacity,
)


class BestFitVMPAlgorithm(BaseAlgorithm):
    """Best Fit algorithm for the VMP problem.
    It is an online algorithm that tries to allocate each virtual machine to the server that will have the least free capacity after the allocation.
    """

    def initialize_algorithm(self, problem: VMPlacementProblem) -> None:
        self.problem = problem

    def run_one_iteration(self) -> Tuple[bool, Dict[int, int]]:

        # Get the virtual machines requirements and the servers capacities
        vm_index_to_vm_requirements = self.problem.get_vm_index_to_vm_requirements()
        server_index_to_server_capacity = (
            self.problem.get_server_index_to_server_capacity()
        )

        # Get the max capacities as well as the averaged normalized capacities
        name_ressource_to_max_capacity = get_name_ressource_to_max_capacity(
            server_index_to_server_capacity=server_index_to_server_capacity,
            ressources=self.problem.ressources,
        )
        server_index_to_averaged_normalized_capacity = (
            get_server_averaged_normalized_capacity(
                server_index_to_server_capacity=server_index_to_server_capacity,
                ressources=self.problem.ressources,
            )
        )

        # Assign each VM to the server that will have the least free (averaged normalized) capacity after the allocation.
        vm_index_to_server_idx: Dict[int, int] = {}
        for j, vm_requirements in vm_index_to_vm_requirements.items():
            best_server_index = None
            lowest_current_capacity = float("inf")
            for i, server_capacities in server_index_to_server_capacity.items():
                # Check first if the server can host the VM
                if all(
                    server_capacities[k] >= vm_requirements[k]
                    for k in vm_requirements.keys()
                ):
                    # Check then if the server has the lowest current capacity
                    if (
                        server_index_to_averaged_normalized_capacity[i]
                        < lowest_current_capacity
                    ):
                        best_server_index = i
                        lowest_current_capacity = (
                            server_index_to_averaged_normalized_capacity[i]
                        )

            # If no server can host the VM, return False
            if best_server_index is None:
                return False, {}

            # Otherwise, update the server capacities, the server average normalized capacities and the mapping from VM index to server index
            else:
                # Update the mapping from VM index to server index
                vm_index_to_server_idx[j] = best_server_index
                
                # Update the server capacities
                for ressource_name in vm_requirements.keys():
                    server_index_to_server_capacity[best_server_index][
                        ressource_name
                    ] -= vm_requirements[ressource_name]

                # Update the server average normalized capacities
                averaged_normalized_vm_requirements = sum(
                    [
                        vm_requirements[ressource_name]
                        / name_ressource_to_max_capacity[ressource_name]
                        for ressource_name in vm_requirements.keys()
                    ]
                ) / len(vm_requirements)
                server_index_to_averaged_normalized_capacity[
                    best_server_index
                ] -= averaged_normalized_vm_requirements

        # Create the solution as a dictionnary of variable index to value
        solution = {}
        for i in range(self.problem.n_servers):
            for j in range(self.problem.n_vms):
                variable_index = self.problem.variable_name_to_variable_index(
                    f"x_{i}_{j}"
                )
                if vm_index_to_server_idx[j] == i:
                    solution[variable_index] = 1
                else:
                    solution[variable_index] = 0
        return True, solution

    def stop_algorithm(self) -> bool:
        return True
