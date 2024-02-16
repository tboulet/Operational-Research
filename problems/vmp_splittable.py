from collections import defaultdict
import itertools
import sys
from typing import Dict, List, Tuple, Union
import numpy as np
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem

import pprint
pp = pprint.PrettyPrinter(indent=4)

class VMPlacementProblemSplittable(VMPlacementProblem):
    """A variant of the VM placement problem with relaxation of the constraint of placing a VM on a single server.
    In this variant, the VMs can be split on up to k servers, providing each load on the servers is at least alpha.
    """

    def __init__(self, config: dict) -> None:
        """Create an instance of the VMPlacementProblemSplittable class.

        It calls the parent constructor which will intialize the problem with the basic VM placement constraints and
        generate the linear formulation of the problem.
        It then relax the integer constraints on the variables, introduce new binary variables z_i_j for each VM j and server i (which represents whether x_i_j is non-zero).
        It finally adds the constraints to ensure that the load on each server is at least alpha and that the VMs are placed on at most k servers.
        """
        self.splittable: List[List[int]] = config.pop("splittable")
        self.n_servers_splittable_max = config.pop("n_servers_splittable_max")
        self.proportion_splittable_min = config.pop("proportion_splittable_min")

        if self.proportion_splittable_min <= 0:
            self.alpha = sys.float_info.epsilon
        else:
            self.alpha = self.proportion_splittable_min

        super().__init__(config)

    def variable_name_to_variable_index(self, variable_name: str) -> int:
        """Convert the name of a variable to its index in the linear formulation.

        Args:
            variable_name (str): the name of the variable

        Returns:
            int: the index of the variable in the linear formulation
        """
        if variable_name.startswith("x"):
            i, j = variable_name.split("_")[1:]
            return int(i) * self.n_vms + int(j)
        elif variable_name.startswith("y"):
            return self.n_vms * self.n_servers + int(variable_name.split("_")[1])
        elif variable_name.startswith("z"):
            i, j = variable_name.split("_")[1:]
            return (
                self.n_vms * self.n_servers
                + self.n_servers
                + int(i) * self.n_vms
                + int(j)
            )
        else:
            raise ValueError(f"Unknown variable name {variable_name}")

    def variable_index_to_variable_name(self, variable_index: int) -> str:
        """Convert the index of a variable to its name in the linear formulation.

        Args:
            variable_index (int): the index of the variable in the linear formulation

        Returns:
            str: the name of the variable
        """
        if variable_index < self.n_vms * self.n_servers:
            i = variable_index // self.n_vms
            j = variable_index % self.n_vms
            return f"x_{i}_{j}"
        elif variable_index < self.n_vms * self.n_servers + self.n_servers:
            return f"y_{variable_index - self.n_vms * self.n_servers}"
        else:
            i = (
                variable_index - self.n_vms * self.n_servers - self.n_servers
            ) // self.n_vms
            j = (
                variable_index - self.n_vms * self.n_servers - self.n_servers
            ) % self.n_vms
            return f"z_{i}_{j}"

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
        """Compute the linear formulation of the splittable VMP problem.

        Args:
            relax_unique_assignment_constraint (bool): whether to have an inequality constraint for the 'each VM j must be assigned' constraint : sum_i x_i_j <= 1 if True sum_i x_i_j = 1 if False

        Returns:
            Dict[int, str]: A dictionary that maps each variable index to its name.
            np.ndarray: The inequality matrix A.
            np.ndarray: The inequality vector b.
            np.ndarray: The equality matrix A_eq.
            np.ndarray: The equality vector b_eq.
            Dict[int, str]: A dictionary that maps each variable index to its type (either "integer" or "continuous").
            Dict[int, Tuple[float, float]]: A dictionary that maps each variable index to its lower and upper bounds.
        """
        self.n_variables = (
            self.n_vms * self.n_servers + self.n_servers + self.n_vms * self.n_servers
        )

        variable_names = {}
        A = []
        b = []
        A_eq = []
        b_eq = []
        variable_types = {}
        variable_bounds = {}

        # Create the variable names, types and bounds
        for i in range(self.n_servers):
            for j in range(self.n_vms):
                variable_index = self.variable_name_to_variable_index(f"x_{i}_{j}")
                variable_names[variable_index] = f"x_{i}_{j}"
                variable_types[variable_index] = "continuous"
                variable_bounds[variable_index] = (0, 1)
        for i in range(self.n_servers):
            variable_index = self.variable_name_to_variable_index(f"y_{i}")
            variable_names[variable_index] = f"y_{i}"
            variable_types[variable_index] = "integer"
            variable_bounds[variable_index] = (0, 1)
        for i in range(self.n_servers):
            for j in range(self.n_vms):
                variable_index = self.variable_name_to_variable_index(f"z_{i}_{j}")
                variable_names[variable_index] = f"z_{i}_{j}"
                variable_types[variable_index] = "integer"
                variable_bounds[variable_index] = (0, 1)

        # Create the cost vector
        c = np.zeros(self.n_variables)
        for i in range(self.n_servers):
            c[self.variable_name_to_variable_index(f"y_{i}")] = 1

        # Constraint : each VM j must be assigned
        for j in range(self.n_vms):
            A_j = [0] * self.n_variables
            for i in range(self.n_servers):
                A_j[self.variable_name_to_variable_index(f"x_{i}_{j}")] = -1
            # Add the inequality constraint or the equality constraint
            if relax_unique_assignment_constraint:
                A.append(A_j)
                b.append(-1)
            else:
                A_eq.append(A_j)
                b_eq.append(-1)

        # Constraint : the sum of the requirement for ressource k of VM of server i must be inferior to the capacity of server i for ressource k
        for i in range(self.n_servers):
            for ressource_name in self.ressources.keys():
                A_ik = [0] * self.n_variables
                for j in range(self.n_vms):
                    A_ik[self.variable_name_to_variable_index(f"x_{i}_{j}")] = (
                        self.vm_index_to_vm_requirements[j][ressource_name]
                    )
                A_ik[self.variable_name_to_variable_index(f"y_{i}")] = (
                    -self.server_index_to_server_capacity[i][ressource_name]
                )
                A.append(A_ik)
                b.append(0)

        # Constraint : the z_i_j variable is defined as 1 if x_i_j is non-zero and 0 otherwise
        for i in range(self.n_servers):
            for j in range(self.n_vms):
                # Constraint 1 : z_i_j >= x_i_j (if x_i_j is non-zero then z_i_j must be 1)
                # x_i_j - z_i_j <= 0
                row = [0] * self.n_variables
                row[self.variable_name_to_variable_index(f"x_{i}_{j}")] = 1
                row[self.variable_name_to_variable_index(f"z_{i}_{j}")] = -1
                A.append(row)
                b.append(0)
                # Constraint 2 : z_i_j * alpha <= x_i_j (if x_i_j is zero then z_i_j must be 0), and additionally if x_i_j is non-zero, it must be at least alpha
                # z_i_j * alpha - x_i_j <= 0
                row = [0] * self.n_variables
                row[self.variable_name_to_variable_index(f"x_{i}_{j}")] = -1
                row[self.variable_name_to_variable_index(f"z_{i}_{j}")] = self.alpha
                A.append(row)
                b.append(0)

        return (
            variable_names,
            c,
            np.array(A),
            np.array(b),
            np.array(A_eq),
            np.array(b_eq),
            variable_types,
            variable_bounds,
            "minimize",
        )

    def apply_solution(
        self, solution: Dict[int, Union[int, float]]
    ) -> Union[bool, float]:
        """Apply the solution to the problem and print the results.

        Args:
            solution (Dict[int, Union[int, float]]): the solution to apply, i.e. a mapping of variable indices to their values

        Returns:
            bool: whether the solution is feasible
            float: the objective value of the solution
        """
        print("\nApplying solution to the VM placement problem :")
        used_servers: Dict[
            int, Dict[str, Union[int, float, List[Tuple[float, int]]]]
        ] = {}
        assignment_percentage_vm = {j: 0 for j in range(self.n_vms)}
        isFeasible = True

        # Check if the solution is feasible and compute the used servers and ressources
        for j in range(self.n_vms):
            for i in range(self.n_servers):
                x_i_j = solution[self.variable_name_to_variable_index(f"x_{i}_{j}")]
                if x_i_j > 10 ** -6 and x_i_j < self.alpha:
                    isFeasible = False
                    if self.verbose >= 1:
                        print(
                            f"Non feasability : VM {j} is placed on server {i} with a load of {x_i_j} which is non null and less than alpha"
                        )
                if x_i_j > 0:
                    # Add the host to the used_hosts dictionary if it is not already added
                    if i not in used_servers:
                        used_servers[i] = self.server_index_to_server_capacity[i].copy()
                        used_servers[i]["assigned_vms"] = []
                    # Remove the resources of the VM from the host and add the VM to the list of assigned VMs
                    used_servers[i]["assigned_vms"].append((x_i_j, j))
                    for resource in self.ressources.keys():
                        used_servers[i][resource] -= (
                            self.vm_index_to_vm_requirements[j][resource] * x_i_j
                        )
                        if used_servers[i][resource] < 0:
                            isFeasible = False
                    # Remove the VM from the list of unassigned VMs
                    assignment_percentage_vm[j] += x_i_j
        if any(assignment_percentage_vm[j] < 1 - 10 ** -6 for j in range(self.n_vms)):
            isFeasible = False
            if self.verbose >= 1:
                print(
                    f"Non feasability : some VMs are not fully placed among the servers"
                )

        # Print the results
        if self.verbose >= 1:
            for i, host_data in used_servers.items():
                print(f"Host {i} is used")
                vm_assignment_repr = ''
                for x_i_j, j in host_data["assigned_vms"]:
                    if x_i_j == 1:
                        vm_assignment_repr += f"{j}, "
                    else:
                        vm_assignment_repr += f"{j}({round(100 * x_i_j)}%), "
                print(f"  - Assigned VMs: {vm_assignment_repr}")
                for resource, value in host_data.items():
                    if resource not in ["host_id", "assigned_vms"]:
                        print(
                            f"  - {resource} used: {self.server_index_to_server_capacity[i][resource] - value}/{self.server_index_to_server_capacity[i][resource]}"
                        )
            print("VM assignment percentage : ")
            pp.pprint(assignment_percentage_vm)
            print(
                "Upper bound on the objective value was: ",
                self.n_servers_solution_non_null,
            )
        return isFeasible, len(used_servers)
