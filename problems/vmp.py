from collections import defaultdict
from typing import Dict, List, Tuple, Union
import numpy as np
from problems.base_problem import BaseOptimizationProblem


class VMPlacementProblem(BaseOptimizationProblem):
    """Class for the VM placement problem."""

    def __init__(self, config) -> None:
        """Create an instance of the VMPlacementProblem class.
        It generate a solvable problem of VM placement.
        To unsure the problem is solvable, it does the following :
        - generate the m VMs
        - group those VMs in n-k groups, with k >= n//2
        - generate n-k servers (the optimal servers) whose capacities are the sum of the requirements of those matrix, plus a margin
        - generate k other servers whose capacities are slightly inferior to the capacities of the optimal servers

        This unsure that :
        - the problem is always solvable (by using the initial configuration of the optimal servers)
        - the optimal solution's value is at most n-k (the feasible optimal server is a lower bound on the optimal solution)
        - because the k other servers are of inferior capacities, they are always worst to pick, so the optimal solution is also superior n-k
        - this ensures that the optimal solution value is n-k, if the margin is sufficiently small

        Args:
            config (_type_): _description_
        """
        super().__init__(config)
        self.n_servers = config["n_servers"]
        self.n_vms = config["n_vms"]
        self.n_servers_solution = config["n_servers_solution"]
        assert (
            self.n_servers_solution <= self.n_servers
        ), "n_servers_solution must be inferior or equal to n_servers"
        self.ressources: Dict[int, Dict[str, Union[str, float, int]]] = config[
            "ressources"
        ]
        self.capacity_bonus_servers_solution_frac: float = config[
            "capacity_bonus_servers_solution_frac"
        ]
        self.capacity_malus_other_servers_frac: float = config[
            "capacity_malus_other_servers_frac"
        ]
        self.relax_unique_assignment_constraint: bool = config[
            "relax_unique_assignment_constraint"
        ]
        self.verbose: int = config["verbose"]

        if self.verbose >= 1:
            print("\nGenerating a VM placement problem...")

        # Generate the VMs
        vm_index_to_vm_requirements = self.sample_vms_data(
            n_vms=self.n_vms, ressources=self.ressources
        )
        if self.verbose >= 1:
            print("VMs requirements : ", vm_index_to_vm_requirements)

        # Construct the first n_servers_solution servers by assigning VMs to them
        server_solution_index_to_server_capacity = (
            self.compute_optimal_server_data_from_vm_data(
                n_servers_solution=self.n_servers_solution,
                vm_index_to_vm_requirements=vm_index_to_vm_requirements,
            )
        )
        if self.verbose >= 2:
            print(
                "Optimal servers capacities : ",
                server_solution_index_to_server_capacity,
            )

        # Add a small bonus to the capacities of the optimal servers
        server_solution_index_to_server_capacity = self.multiply_randomly_server_capacities(
            server_index_to_server_capacity=server_solution_index_to_server_capacity,
            max_capacity_multiplier=1 + self.capacity_bonus_servers_solution_frac,
            min_capacity_multiplier=1,
        )
        if self.verbose >= 2:
            print(
                "Optimal servers capacities with bonus : ",
                server_solution_index_to_server_capacity,
            )

        # Construct the last n_servers-n_servers_solution servers by assigning creating "inferior copies" of the first servers
        server_other_index_to_server_capacity = {}
        for i in range(self.n_servers_solution, self.n_servers):
            i_serv_solution = np.random.randint(self.n_servers_solution)
            server_other_index_to_server_capacity[i] = (
                server_solution_index_to_server_capacity[i_serv_solution].copy()
            )
        if self.verbose >= 2:
            print("Other servers capacities : ", server_other_index_to_server_capacity)

        # Add a malus to the capacities of the other servers
        server_other_index_to_server_capacity = (
            self.multiply_randomly_server_capacities(
                server_index_to_server_capacity=server_other_index_to_server_capacity,
                max_capacity_multiplier=1,
                min_capacity_multiplier=1 - self.capacity_malus_other_servers_frac,
            )
        )
        if self.verbose >= 2:
            print(
                "Other servers capacities with malus : ",
                server_other_index_to_server_capacity,
            )

        # Save the data
        self.vm_index_to_vm_requirements = vm_index_to_vm_requirements
        self.server_index_to_server_capacity = {
            **server_solution_index_to_server_capacity,
            **server_other_index_to_server_capacity,
        }

        # Compute the optimal solution value (can be only an upper bound if malus too low or bonus too high)
        self.n_servers_solution_non_null = (
            sum(  # count the number of solution server will non fully empty capacities.
                [
                    1
                    for i in range(self.n_servers_solution)
                    if sum(self.server_index_to_server_capacity[i].values()) > 0
                ]
            )
        )
        if self.verbose >= 1:
            print("Servers capacities : ", self.server_index_to_server_capacity)
            print(
                "Optimal solution value (or upper bound) : ",
                self.n_servers_solution_non_null,
            )

        # Compute linear formulation
        (
            self.variable_names,
            self.c,
            self.A,
            self.b,
            self.A_eq,
            self.b_eq,
            self.variable_types,
            self.variable_bounds,
            self.optimization_sense,
        ) = self.compute_linear_formulation(
            relax_unique_assignment_constraint=self.relax_unique_assignment_constraint,
        )
        if self.verbose >= 2:
            print("\nLinear formulation computed.")
            print("Variable names : ", self.variable_names)
            print("c : ", self.c)
            print("A : ", self.A)
            print("b : ", self.b)
            print("A_eq : ", self.A_eq)
            print("b_eq : ", self.b_eq)
            print("Variable types : ", self.variable_types)
            print("Variable bounds : ", self.variable_bounds)

    def sample_vms_data(
        self, n_vms: int, ressources: Dict[str, Dict[str, int]]
    ) -> Dict[int, Dict[str, int]]:
        """Create randomly a set of VMs with their requirements.

        Args:
            n_vms (int): the number of VMs to create
            ressources (Dict[str, Dict[str, int]]): a mapping of ressource names to their distribution parameters

        Returns:
            Dict[int, Dict[str, int]]: the mapping of VMs to their requirements
        """
        vm_index_to_vm_requirements: Dict[int, Dict[str, int]] = {}
        for j in range(self.n_vms):
            vm_index_to_vm_requirements[j] = {}
            for ressource_name in self.ressources.keys():
                vm_index_to_vm_requirements[j][ressource_name] = np.random.randint(
                    self.ressources[ressource_name]["vm_min"],
                    self.ressources[ressource_name]["vm_max"],
                )
        return vm_index_to_vm_requirements

    def compute_optimal_server_data_from_vm_data(
        self,
        n_servers_solution: int,
        vm_index_to_vm_requirements: Dict[int, Dict[str, int]],
    ) -> Dict[int, Dict[str, int]]:
        """Create the server data of an 'optimal' solution from the VM data, by creating server as sum of the VMs requirements.

        Args:
            vm_index_to_vm_requirements (Dict[int, Dict[str, int]]): the mapping of VMs to their requirements

        Returns:
            Dict[int, Dict[str, int]]: the mapping of servers to their capacities
        """
        ressources_names = list(vm_index_to_vm_requirements[0].keys())
        n_vms = len(vm_index_to_vm_requirements)

        # Create n_servers_solution empty servers
        server_index_to_server_capacity: Dict[int, Dict[str, int]] = {
            i: {ressource_name: 0 for ressource_name in ressources_names}
            for i in range(n_servers_solution)
        }

        # Randomly add each VM requirements to a server
        for j in range(n_vms):
            i = np.random.randint(n_servers_solution)
            for ressource_name in ressources_names:
                server_index_to_server_capacity[i][
                    ressource_name
                ] += vm_index_to_vm_requirements[j][ressource_name]
        return server_index_to_server_capacity

    def multiply_randomly_server_capacities(
        self,
        server_index_to_server_capacity: Dict[int, Dict[str, float]],
        max_capacity_multiplier: float,
        min_capacity_multiplier: float,
    ) -> Dict[int, Dict[str, float]]:
        """Randomly multiply the capacities of the servers by a factor between min_capacity_multiplier and max_capacity_multiplier.

        Args:
            server_index_to_server_capacity (Dict[int, Dict[str, int]]): the mapping of servers to their capacities
            max_capacity_multiplier (float): the maximum multiplier to apply to the capacities
            min_capacity_multiplier (float): the minimum multiplier to apply to the capacities

        Returns:
            Dict[int, Dict[str, int]]: the mapping of servers to their capacities
        """
        if len(server_index_to_server_capacity) == 0:
            return server_index_to_server_capacity
        ressource_names = next(iter(server_index_to_server_capacity.values())).keys()
        for i in server_index_to_server_capacity.keys():
            for ressource_name in ressource_names:
                server_index_to_server_capacity[i][ressource_name] = int(
                    server_index_to_server_capacity[i][ressource_name]
                    * np.random.uniform(
                        min_capacity_multiplier, max_capacity_multiplier
                    )
                )
        return server_index_to_server_capacity

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
        else:
            return f"y_{variable_index - self.n_vms * self.n_servers}"

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
        """Compute the linear formulation of the problem.

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
        self.n_variables = self.n_vms * self.n_servers + self.n_servers

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
                variable_types[variable_index] = "integer"
                variable_bounds[variable_index] = (0, 1)
        for i in range(self.n_servers):
            variable_index = self.variable_name_to_variable_index(f"y_{i}")
            variable_names[variable_index] = f"y_{i}"
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

    def get_linear_formulation(
        self,
    ) -> Tuple[
        Dict[int, str],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[int, str],
        Dict[int, Tuple[float, float]],
        str,
    ]:
        return (
            self.variable_names,
            self.c,
            self.A,
            self.b,
            self.A_eq,
            self.b_eq,
            self.variable_types,
            self.variable_bounds,
            self.optimization_sense,
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
        used_servers: Dict[int, Dict[str, Union[int, float, List[int]]]] = {}
        unassigned_vms = set(range(self.n_vms))
        isFeasible = True

        # Check if the solution is feasible and compute the used servers and ressources
        for j in range(self.n_vms):
            for i in range(self.n_servers):
                if solution[self.variable_name_to_variable_index(f"x_{i}_{j}")] == 1:
                    # Add the host to the used_hosts dictionary if it is not already added
                    if i not in used_servers:
                        used_servers[i] = self.server_index_to_server_capacity[i].copy()
                        used_servers[i]["assigned_vms"] = []
                    # Remove the resources of the VM from the host and add the VM to the list of assigned VMs
                    used_servers[i]["assigned_vms"].append(j)
                    for resource in self.ressources.keys():
                        used_servers[i][resource] -= self.vm_index_to_vm_requirements[
                            j
                        ][resource]
                        if used_servers[i][resource] < 0:
                            isFeasible = False
                    # Remove the VM from the list of unassigned VMs
                    unassigned_vms.discard(j)
        if len(unassigned_vms) > 0:
            isFeasible = False

        # Print the results
        if self.verbose >= 1:
            for i, host_data in used_servers.items():
                print(f"Host {i} is used")
                print(f"  - Assigned VMs: {host_data['assigned_vms']}")
                for resource, value in host_data.items():
                    if resource not in ["host_id", "assigned_vms"]:
                        print(
                            f"  - {resource} used: {self.server_index_to_server_capacity[i][resource] - value}/{self.server_index_to_server_capacity[i][resource]}"
                        )
            print("Unassigned VMs: ", unassigned_vms)
            print(
                "Upper bound on the objective value was: ",
                self.n_servers_solution_non_null,
            )
        return isFeasible, len(used_servers)
