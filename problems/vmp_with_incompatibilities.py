from collections import defaultdict
import itertools
from typing import Dict, List, Tuple, Union
import numpy as np
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem


class VMPlacementProblemWithIncomp(VMPlacementProblem):
    """A variant of the VM placement problem with additional incompatibilities constraints.
    Some VMs cannot be placed on the same server as some other VMs, for example because one is the safety backup of the other.
    """

    def __init__(self, config : dict) -> None:
        """Create an instance of the VMPlacementProblemWithIncomp class.
        
        It calls the parent constructor which will intialize the problem with the basic VM placement constraints and
        generate the linear formulation of the problem.
        It then adds the incompatibility constraints to the linear formulation.
        """
        assert "incomp_vm_indexes" in config, "The incomp_vm_indexes argument is missing from the config"
        config_vmp_with_family = config.copy()
        self.incomp_vm_indexes : List[List[int]] = config_vmp_with_family.pop("incomp_vm_indexes")
        super().__init__(config=config_vmp_with_family)
        if len(self.incomp_vm_indexes) == 0:
            return
        A_incomp = []
        b_incomp = []
        for indexes_incomp_vm in self.incomp_vm_indexes:
            for j1, j2 in itertools.combinations(indexes_incomp_vm, 2):
                assert j1 != j2, f"Two same VMs cannot be incompatibles, but {j1} was found to be incompatibles with itself"
                assert j1 in range(self.n_vms), f"VM index {j1} is not valid"
                assert j2 in range(self.n_vms), f"VM index {j2} is not valid"
                # Incompatibility constraints: j1 and j2 cannot be placed on the same server
                # This translates to : for all server i, x_i_j1 + x_i_j2 <= 1
                for i in range(self.n_servers):
                    row = [0] * self.n_variables
                    row[self.variable_name_to_variable_index(f"x_{i}_{j1}")] = 1
                    row[self.variable_name_to_variable_index(f"x_{i}_{j2}")] = 1
                    A_incomp.append(row)
                    b_incomp.append(1)
        A_incomp = np.array(A_incomp)
        b_incomp = np.array(b_incomp)
        self.A = np.vstack([self.A, A_incomp])
        self.b = np.concatenate([self.b, b_incomp])

    def apply_solution(
        self, solution: Dict[int, Union[int, float]]
    ) -> Union[bool, float]:
        """Apply the solution to the problem and print the results.
        This method first use the parent method to see if the solution with the incompatible constraints relaxed is feasible.
        If it is, it then check if the solution is feasible with the incompatible constraints.
        
        Args:
            solution (Dict[int, Union[int, float]]): the solution to apply, i.e. a mapping of variable indices to their values

        Returns:
            bool: whether the solution is feasible
            float: the objective value of the solution
        """
        is_feasible_relaxed, objective_value_relaxed = super().apply_solution(solution)
        if not is_feasible_relaxed:
            return False, objective_value_relaxed
        else:
            # Check if the solution is feasible with the incompatible constraints
            for indexes_incomp_vm in self.incomp_vm_indexes:
                for j1, j2 in itertools.combinations(indexes_incomp_vm, 2):
                    for i in range(self.n_servers):
                        if solution[self.variable_name_to_variable_index(f"x_{i}_{j1}")] + solution[self.variable_name_to_variable_index(f"x_{i}_{j2}")] > 1:
                            is_feasible_relaxed = False
                            print(f"Incompatible VMs {j1} and {j2} are placed on the same server {i}")
            return is_feasible_relaxed, objective_value_relaxed
