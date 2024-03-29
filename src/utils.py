from re import U
import sys
from typing import Dict, Union

import numpy as np

EPSILON = 1e-6

def to_integer(x: Union[int, float, str, None]) -> Union[int, None]:
    if isinstance(x, int) or isinstance(x, float):
        return int(x)
    elif x == "inf":
        return sys.maxsize
    elif x == "-inf":
        return -sys.maxsize
    elif isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            raise ValueError(f"Cannot convert {x} to integer, please specify something like '2' or '3.0' or 'inf'.")
    elif x is None:
        return None
    else:
        raise ValueError(f"Cannot convert {x} to numeric")


def try_get_seed(config: Dict) -> int:
    """Will try to extract the seed from the config, or return a random one if not found

    Args:
        config (Dict): the run config

    Returns:
        int: the seed
    """
    try:
        seed = config["seed"]
        if not isinstance(seed, int):
            seed = np.random.randint(0, 1000)
    except KeyError:
        seed = np.random.randint(0, 1000)
    return seed