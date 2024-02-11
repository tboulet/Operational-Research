from typing import Dict, Union

import pyomo.environ as pyo

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem


class GPLK_Solver(BaseAlgorithm):
    """A solver that uses Pyomo and the GPLK solver to solve any MILP problem.
    This algorithm provides probably a lower bound on any
    """

    def initialize_algorithm(self, problem: BaseOptimizationProblem) -> None:
        self.problem = problem

    def run_one_iteration(self) -> Dict[int, Union[int, float]]:
        (
            variable_names,
            c,
            A,
            b,
            A_eq,
            b_eq,
            variable_types,   # TODO : include this in the problem
            variable_bounds,
            sense,
        ) = self.problem.get_linear_formulation()

        variable_idx_to_name = {
            i: variable_names[i] for i in range(len(variable_names))
        }
        variable_name_to_idx = {
            variable_names[i]: i for i in range(len(variable_names))
        }
        assert sense in ["minimize", "maximize"], f"Unknown sense {sense}"

        # Create a concrete model
        model = pyo.ConcreteModel()

        # Define the decision variables
        model.x = pyo.Var(variable_names, domain=pyo.Binary)

        # Define the objective function
        model.profit = pyo.Objective(
            expr=sum(
                model.x[variable_name_to_idx[variable_names[i]]] * c[i]
                for i in range(len(variable_names))
            ),
            sense=1 if sense == "minimize" else -1,
        )

        # Define the constraints
        model.constraints = pyo.ConstraintList()
        for i in range(A.shape[0]):
            model.constraints.add(
                sum(
                    model.x[variable_name_to_idx[variable_names[j]]] * A[i, j]
                    for j in range(len(variable_names))
                )
                <= b[i]
            )
        for i in range(A_eq.shape[0]):
            model.constraints.add(
                sum(
                    model.x[variable_name_to_idx[variable_names[j]]] * A_eq[i, j]
                    for j in range(len(variable_names))
                )
                == b_eq[i]
            )

        solver = pyo.SolverFactory("glpk")
        results = solver.solve(model)

        return {
            variable_name_to_idx[variable_names[i]]: model.x[
                variable_name_to_idx[variable_names[i]]
            ].value
            for i in range(len(variable_names))
        }

    def stop_algorithm(self) -> bool:
        return True
