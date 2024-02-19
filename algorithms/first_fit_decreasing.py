from typing import Dict, Tuple, Union

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem


class FirstFitDecreasingVMPAlgorithm(BaseAlgorithm):
    """First Fit algorithm for the VMP problem, but first sort the server by decreasing average normalized capacity.
    It is a simple algorithm that tries to allocate each virtual machine to the first server that can host it.
    """

    def initialize_algorithm(self, problem: VMPlacementProblem) -> None:
        self.problem = problem

    def run_one_iteration(self) -> Tuple[bool, Dict[int, int]]:
        vm_index_to_vm_requirements = self.problem.get_vm_index_to_vm_requirements()
        server_index_to_server_capacity = (
            self.problem.get_server_index_to_server_capacity()
        )

        # Sort the server by decreasing average normalized capacity
        name_ressource_to_max_capacity = {
            name: max(
                [
                    server_capacities[name]
                    for server_capacities in server_index_to_server_capacity.values()
                ]
            )
            for name in self.problem.ressources
        }
        server_index_to_server_normalized_capacity = {
            server_index: {
                name: server_capacities[name] / name_ressource_to_max_capacity[name]
                for name in self.problem.ressources
            }
            for server_index, server_capacities in server_index_to_server_capacity.items()
        }
        server_index_to_server_averaged_normalized_capacity = {
            server_index: sum(server_capacities.values()) / len(server_capacities)
            for server_index, server_capacities in server_index_to_server_normalized_capacity.items()
        }
        list_server_items_sorted_by_decreasing_average_normalized_capacity = sorted(
            server_index_to_server_capacity.items(),
            key=lambda server_index_to_server_capacity: server_index_to_server_averaged_normalized_capacity[
                server_index_to_server_capacity[0]
            ],
            reverse=True,
        )

        vm_index_to_server_idx: Dict[int, int] = {}
        for j, vm_requirements in vm_index_to_vm_requirements.items():
            for (
                i,
                _,
            ) in list_server_items_sorted_by_decreasing_average_normalized_capacity:
                server_capacities = server_index_to_server_capacity[i]
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
