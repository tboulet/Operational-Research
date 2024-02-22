from typing import Dict, Tuple, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem


class FirstFitVMPAlgorithm(BaseAlgorithm):
    """First Fit algorithm for the VMP problem.
    It is a simple algorithm that tries to allocate each virtual machine to the first server that can host it.
    """

    def initialize_algorithm(self, problem: VMPlacementProblem) -> None:
        self.problem = problem

    def run_one_iteration(self) -> Tuple[bool, Dict[int, int]]:

        # Get the virtual machines requirements and the servers capacities
        vm_index_to_vm_requirements = self.problem.get_vm_index_to_vm_requirements()
        server_index_to_server_capacity = (
            self.problem.get_server_index_to_server_capacity()
        )

        # Try to assign each virtual machine to the first server that can host it
        vm_index_to_server_idx: Dict[int, int] = (
            {}
        )  # mapping from vm index to server index
        for j, vm_requirements in vm_index_to_vm_requirements.items():
            for i, server_capacities in server_index_to_server_capacity.items():
                if all(
                    server_capacities[k] >= vm_requirements[k]
                    for k in vm_requirements.keys()
                ):
                    vm_index_to_server_idx[j] = i
                    for ressource_name in vm_requirements.keys():
                        server_capacities[ressource_name] -= vm_requirements[
                            ressource_name
                        ]
                    break
            else:
                return False, {}

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
