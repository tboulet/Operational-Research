from queue import PriorityQueue
from typing import Dict, Tuple, Union, List, Optional

from algorithms.base_algorithm import BaseAlgorithm
from problems.base_problem import BaseOptimizationProblem
from problems.vmp import VMPlacementProblem
from scipy.optimize import linprog

class OptimizationProblemNode:
    """This class represents a node in the B&B tree, i.e. the original Integer Programming minimization problem with additional constraints on some of the decision variables.
    It contains the complete definition of a Integer Programming problem (c, A, b, bounds) as well as the value and solution of its linear relaxation.
    """
    def __init__(self, 
            c : List[float],
            A : List[List[float]],
            b : List[float],
            bounds : List[Tuple[Optional[float], Optional[float]]],
            ):
        """Create a new problem node from the definition of an Integer Programming problem, and compute the solution of its linear relaxation.

        Args:
            c (List[float]): the coefficients of the objective function
            A (List[List[float]]): the coefficients of the inequality constraints, in the form of a matrix
            b (List[float]): the right-hand side of the inequality constraints
            bounds (List[Tuple[int, int]]): the bounds of the integer decision variables, e.g. (0, 1) for binary variables or (0, None) for non-negative variables
        """
        # Define the (sub?)-problem
        self.c = c
        self.A = A
        self.b = b
        self.bounds = bounds
        # Solve the linear relaxation of the problem
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        self.success = res.success
        if not self.success:
            self.lower_bound, self.x_relaxed_optimal = float('inf'), None
        else:
            self.lower_bound = res.fun
            self.x_relaxed_optimal = [round(x) if abs(round(x) - x) < 1e-5 else x for x in res.x] # Round the relaxed optimal solution to the nearest integer if they are close to an integer
     
    def __lt__(self, other : 'OptimizationProblemNode'):
        """This method is implemented in case of equality in the priority queue, which happens in case of equal lower bounds. The results does not really matter."""
        return self.lower_bound < other.lower_bound
    

class ChildProblemNode(OptimizationProblemNode):
    """This class creates a Problem Node from a parent node and an additional unequality constraint on one of the decision variables."""
    def __init__(self, 
            parent_problem : OptimizationProblemNode, 
            index_variable : int, 
            value : float, 
            sense : str,
            ):
        """Create a new problem node from a parent node and an additional constraint on one of the decision variables.

        Args:
            parent_problem (ProblemNode): the parent problem node
            index_variable (int): the index of the decision variable to which the constraint is applied
            value (float): the value of the decision variable in the constraint
            sense (str): the sense of the constraint. It can be either 'inferior' or 'superior' (to value)
        """
        sense_int = 1 if sense == 'inferior' else -1
        super().__init__(
            parent_problem.c, 
            parent_problem.A + [[0] * index_variable + [sense_int] + [0] * (len(parent_problem.x_relaxed_optimal) - index_variable - 1)],
            parent_problem.b + [sense_int * value],
            parent_problem.bounds,
            )
        
        # # Second method, possibly more time efficient, but doesn't work
        # super().__init__(parent_problem.c, parent_problem.A, parent_problem.b, parent_problem.bounds)
        # if sense == 'inferior':
        #     self.bounds[index_variable] = (self.bounds[index_variable][0], min(self.bounds[index_variable][1], value))
        # elif sense == 'superior':
        #     self.bounds[index_variable] = (max(self.bounds[index_variable][0], value), self.bounds[index_variable][1])
        # else:
        #     raise ValueError("The sense parameter must be either 'inferior' or 'superior'")
        
        
def is_integer_solution(x : List[float]) -> bool:
    return all(abs(round(x_i) - x_i) < 1e-5 for x_i in x)        

import numpy as np

def convert_eq_to_ineq_constraints(A, b, A_eq, b_eq) -> Tuple[np.ndarray, np.ndarray]:
    """Convert equality constraints to inequality constraints.

    Args:
        A (np.ndarray): the coefficients of the inequality constraints, in the form of a matrix
        b (np.ndarray): the right-hand side of the inequality constraints
        A_eq (np.ndarray): the coefficients of the equality constraints, in the form of a matrix
        b_eq (np.ndarray): the right-hand side of the equality constraints

    Returns:
        A_new (np.ndarray): the coefficients of the inequality constraints, in the form of a matrix
        b_new (np.ndarray): the right-hand side of the inequality constraints
    """
    # Append A_eq and -A_eq to A
    A_new = np.vstack((A, A_eq, -A_eq))
    # Append b_eq and -b_eq to b
    b_new = np.concatenate((b, b_eq, -b_eq), axis=0)
    return A_new, b_new


class BranchAndBoundAlgorithm(BaseAlgorithm):
    """The Branch And Bound algorithm."""

    def initialize_algorithm(self, problem: VMPlacementProblem) -> None:
        self.problem = problem
        
        # Get the linear formulation
        (
            variable_names,
            self.c,
            self.A,
            self.b,
            self.A_eq,
            self.b_eq,
            self.variable_types,
            self.variable_bounds,
            self.sense,
        ) = self.problem.get_linear_formulation()
        
        # If the problem is a maximization problem, convert it to a minimization problem
        if self.sense == 'maximize':
            self.c = [-x for x in self.c]    
        elif self.sense != 'minimize':
            raise ValueError("The 'sense' parameter must be either 'minimize' or 'maximize'")
        
        # Convert the equality constraints to inequality constraints : Ax=b is equivalent to Ax<=b and Ax>=b
        if len(self.A_eq) > 0:
            self.A, self.b = convert_eq_to_ineq_constraints(self.A, self.b, self.A_eq, self.b_eq)

        # Convert the bounds to the format expected by linprog
        bounds = [self.variable_bounds[i] for i in range(len(self.variable_bounds))]
        
        # Initialize the root node
        self.upper_bound = float('inf')  # Initialize the upper bound to infinity as we have not found any solution yet
        self.best_solution = None
        self.has_found_solution = False
        root = OptimizationProblemNode(self.c, self.A, self.b, bounds)

        # Priority queue to store nodes based on their bounds
        self.priority_queue = PriorityQueue()
        self.priority_queue.put((root.lower_bound, root))
        
                
    def run_one_iteration(self) -> Tuple[bool, Dict[int, int]]:
        assert not self.priority_queue.empty(), "The priority queue is empty. The algorithm should have been stopped."
        _, node = self.priority_queue.get()
        node : OptimizationProblemNode
        
        # If the node is not valid, skip it
        if not node.success:
            return self.has_found_solution, self.best_solution
        
        # If the node's bound is worse than the current best solution, prune the branch
        if node.lower_bound > self.upper_bound:
            return self.has_found_solution, self.best_solution

        # If the node is a complete solution (integer solution), then its lower bound is the function found for that integer solution, so we can update the upper bound and the best solution
        if is_integer_solution(node.x_relaxed_optimal):
            self.upper_bound = node.lower_bound
            self.has_found_solution = True
            self.best_solution = node.x_relaxed_optimal
            return self.has_found_solution, self.best_solution

        # Expand the node and add child nodes to the priority queue
        first_non_integer_variable = next(i for i, x in enumerate(node.x_relaxed_optimal) if abs(round(x) - x) >= 1e-5)
        node_x_smaller = ChildProblemNode(node, first_non_integer_variable, int(node.x_relaxed_optimal[first_non_integer_variable]), 'inferior')
        node_x_bigger = ChildProblemNode(node, first_non_integer_variable, int(node.x_relaxed_optimal[first_non_integer_variable]) + 1, 'superior')
        for child_node in [node_x_smaller, node_x_bigger]:
            # Only add child nodes if their bounds are better than the current best solution
            if child_node.lower_bound < self.upper_bound:
                self.priority_queue.put((child_node.lower_bound, child_node))

        return self.has_found_solution, self.best_solution
    
    def stop_algorithm(self) -> bool:
        return self.priority_queue.empty()
