from typing import Dict, Union

import pyomo.environ as pyo

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem
from src.pyomo_utils import (
    variable_string_sense_to_pyomo_sense,
    variable_string_type_to_pyomo_domain,
)


class PyomoSolver(BaseAlgorithm):
    """A solver that uses Pyomo a solver for MILP problems.
    This algorithm provides probably a lower bound on any solution other algorithms may find.
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
            variable_types,
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
        for variable_idx, variable_name in variable_idx_to_name.items():
            setattr(
                model,
                variable_name,
                pyo.Var(
                    domain=variable_string_type_to_pyomo_domain[
                        variable_types[variable_idx]
                    ],
                    bounds=variable_bounds[variable_idx],
                ),
            )

        # Define the objective function
        model.profit = pyo.Objective(
            expr=sum(
                getattr(model, variable_names[i]) * c[i]
                for i in range(len(variable_names))
            ),
            sense=variable_string_sense_to_pyomo_sense[sense],
        )

        # Define the constraints
        model.constraints = pyo.ConstraintList()
        for i in range(A.shape[0]):
            model.constraints.add(
                sum(
                    getattr(model, variable_names[j]) * A[i, j]
                    for j in range(len(variable_names))
                )
                <= b[i]
            )
        for i in range(A_eq.shape[0]):
            model.constraints.add(
                sum(
                    getattr(model, variable_names[j]) * A_eq[i, j]
                    for j in range(len(variable_names))
                )
                == b_eq[i]
            )

        solver = pyo.SolverFactory(self.config["solver_tag"])
        results = solver.solve(model)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            return False, None
        else:
            return True, {
                variable_name_to_idx[variable_names[i]]: getattr(
                    model, variable_names[i]
                ).value
                for i in range(len(variable_names))
            }

    def stop_algorithm(self) -> bool:
        return True
