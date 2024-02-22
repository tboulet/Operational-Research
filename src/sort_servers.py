from typing import Any, Dict, List, Tuple, Union


def get_name_ressource_to_max_capacity(
    server_index_to_server_capacity: Dict[int, Dict[str, float]],
    ressources: Dict[str, Any],
) -> Dict[str, float]:
    """Return the maximum capacity for each ressource.

    Args:
        server_index_to_server_capacity (Dict[int, Dict[str, float]]): the mapping from server index to server capacity
        ressources (Dict[str, Any]): a dictionary whose keys are the ressources names

    Returns:
        Dict[str, float]: the dictionnary {k : max_i(c_ik) for k in ressources} where c_ik is the capacity of server i for ressource k
    """
    return {
        name: max(
            [
                server_capacities[name]
                for server_capacities in server_index_to_server_capacity.values()
            ]
        )
        for name in ressources
    }


def get_server_averaged_normalized_capacity(
    server_index_to_server_capacity: Dict[int, Dict[str, float]],
    ressources: Dict[str, Any],
) -> Dict[int, float]:
    """Get the server averaged normalized capacity.

    Args:
        server_index_to_server_capacity (Dict[int, Dict[str, float]]): the mapping from server index to server capacity
        ressources (Dict[str, Any]): a dictionnary whose keys are the ressources names

    Returns:
        Dict[int, float]: the server averaged normalized capacity. Formally,
        it returns the dictionnary {i : (1/K) * sum_k(c_ik / max_i(c_ik)) for i in servers} where c_ik is the capacity of server i for ressource k
    """
    # Get the server averaged normalized capacity
    name_ressource_to_max_capacity = get_name_ressource_to_max_capacity(
        server_index_to_server_capacity=server_index_to_server_capacity,
        ressources=ressources,
    )

    server_normalized_capacity = {
        server_index: {
            name: server_capacities[name] / name_ressource_to_max_capacity[name]
            for name in ressources
        }
        for server_index, server_capacities in server_index_to_server_capacity.items()
    }
    server_averaged_normalized_capacity = {
        server_index: sum(server_normalized_capacities.values())
        / len(server_normalized_capacities)
        for server_index, server_normalized_capacities in server_normalized_capacity.items()
    }

    return server_averaged_normalized_capacity


def get_sorted_server_items(
    server_index_to_server_capacity: Dict[int, Dict[str, float]],
    ressources: Dict[str, Any],
) -> List[Tuple[int, Dict[str, float]]]:
    """Sort the server by decreasing average normalized capacity.

    Args:
        server_index_to_server_capacity (Dict[int, Dict[str, float]]): the mapping from server index to server capacity
        ressources (Dict[str, Any]): a dictionnary whose keys are the ressources names

    Returns:
        List[Tuple[int, Dict[str, float]]]: the list of server items sorted by decreasing average normalized capacity
    """
    server_averaged_normalized_capacity = get_server_averaged_normalized_capacity(
        server_index_to_server_capacity=server_index_to_server_capacity,
        ressources=ressources,
    )
    list_server_items_sorted_by_decreasing_average_normalized_capacity = sorted(
        server_index_to_server_capacity.items(),
        key=lambda server_index_to_server_capacity: server_averaged_normalized_capacity[
            server_index_to_server_capacity[0]
        ],
        reverse=True,
    )

    return list_server_items_sorted_by_decreasing_average_normalized_capacity
