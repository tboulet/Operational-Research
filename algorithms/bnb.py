from queue import PriorityQueue
from typing import List, Optional, Tuple, Union
from scipy.optimize import linprog


class OptimizationProblemNode:
    """This class represents a node in the B&B tree, i.e. the original Integer Programming minimization problem with additional constraints on some of the decision variables.
    It contains the complete definition of a Integer Programming problem (c, A, b, bounds) as well as the value and solution of its linear relaxation.
    """

    def __init__(
        self,
        c: List[float],
        A: List[List[float]],
        b: List[float],
        bounds: List[Tuple[Optional[float], Optional[float]]],
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
        res = linprog(
            c,
            A_ub=A if A.size > 0 else None,
            b_ub=b if b.size > 0 else None,
            bounds=bounds,
            method="highs",
        )
        self.success = res.success
        if not self.success:
            self.lower_bound, self.x_relaxed_optimal = float("inf"), None
        else:
            self.lower_bound = res.fun
            self.x_relaxed_optimal = [
                round(x) if abs(round(x) - x) < 1e-5 else x for x in res.x
            ]  # Round the relaxed optimal solution to the nearest integer if they are close to an integer

    def __lt__(self, other: "OptimizationProblemNode"):
        """This method is implemented in case of equality in the priority queue, which happens in case of equal lower bounds. The results does not really matter."""
        return self.lower_bound < other.lower_bound


class ChildProblemNode(OptimizationProblemNode):
    """This class creates a Problem Node from a parent node and an additional unequality constraint on one of the decision variables."""

    def __init__(
        self,
        parent_problem: OptimizationProblemNode,
        index_variable: int,
        value: float,
        sense: str,
    ):
        """Create a new problem node from a parent node and an additional constraint on one of the decision variables.

        Args:
            parent_problem (ProblemNode): the parent problem node
            index_variable (int): the index of the decision variable to which the constraint is applied
            value (float): the value of the decision variable in the constraint
            sense (str): the sense of the constraint. It can be either 'inferior' or 'superior' (to value)
        """
        sense_int = 1 if sense == "inferior" else -1
        super().__init__(
            parent_problem.c,
            parent_problem.A
            + [
                [0] * index_variable
                + [sense_int]
                + [0] * (len(parent_problem.x_relaxed_optimal) - index_variable - 1)
            ],
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


def is_integer_solution(x: List[float]) -> bool:
    return all(abs(round(x_i) - x_i) < 1e-5 for x_i in x)


def branch_and_bound(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    bounds: List[Tuple[Optional[float], Optional[float]]],
    sense: str = "minimize",
) -> Tuple[List[float], float]:
    """Apply the Branch and Bound algorithm to solve an Integer Programming problem.

    Args:
        c (List[float]): the coefficients of the objective function
        A (List[List[float]]): the coefficients of the inequality constraints, in the form of a matrix
        b (List[float]): the right-hand side of the inequality constraints
        bounds (List[Tuple[int, int]]): the bounds of the integer decision variables, e.g. (0, 1) for binary variables or (0, None) for non-negative variables
        sense (str): the sense of the optimization problem. It can be either 'minimize' or 'maximize'

    Returns:
        List[float]: the optimal solution of the problem, or None if the problem is infeasible
        float: the optimal value of the objective function, or None if the problem is infeasible
    """
    # If the problem is a maximization problem, convert it to a minimization problem
    if sense == "maximize":
        c = [-x for x in c]
    elif sense != "minimize":
        raise ValueError(
            "The 'sense' parameter must be either 'minimize' or 'maximize'"
        )

    # Initialize the root node
    upper_bound = float(
        "inf"
    )  # Initialize the upper bound to infinity as we have not found any solution yet
    best_solution = None
    root = OptimizationProblemNode(c, A, b, bounds)

    # Priority queue to store nodes based on their bounds
    priority_queue = PriorityQueue()
    priority_queue.put((root.lower_bound, root))

    # Main loop
    while not priority_queue.empty():
        _, node = priority_queue.get()
        node: OptimizationProblemNode

        # If the node is not valid, skip it
        if not node.success:
            continue

        # If the node's bound is worse than the current best solution, prune the branch
        if node.lower_bound > upper_bound:
            continue

        # If the node is a complete solution (integer solution), then its lower bound is the function found for that integer solution, so we can update the upper bound and the best solution
        if is_integer_solution(node.x_relaxed_optimal):
            upper_bound = node.lower_bound
            best_solution = node.x_relaxed_optimal
            continue

        # Expand the node and add child nodes to the priority queue
        first_non_integer_variable = next(
            i for i, x in enumerate(node.x_relaxed_optimal) if abs(round(x) - x) >= 1e-5
        )
        node_x_smaller = ChildProblemNode(
            node,
            first_non_integer_variable,
            int(node.x_relaxed_optimal[first_non_integer_variable]),
            "inferior",
        )
        node_x_bigger = ChildProblemNode(
            node,
            first_non_integer_variable,
            int(node.x_relaxed_optimal[first_non_integer_variable]) + 1,
            "superior",
        )
        for child_node in [node_x_smaller, node_x_bigger]:
            # Only add child nodes if their bounds are better than the current best solution
            if child_node.lower_bound < upper_bound:
                priority_queue.put((child_node.lower_bound, child_node))

    if best_solution is None:
        return None, None

    if sense == "maximize":
        upper_bound = -upper_bound
    return best_solution, upper_bound
