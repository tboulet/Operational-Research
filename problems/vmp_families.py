from collections import defaultdict
import itertools
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from problems.base_problem import BaseOptimizationProblem
from problems.vmp_with_incompatibilities import VMPlacementProblemWithIncomp


class VMPlacementProblemWithIncompFamilies(VMPlacementProblemWithIncomp):
    """A variant of the VM placement problem where VM are grouped in families and some families cannot be placed on the same server as some other families."""

    def __init__(self, config: dict) -> None:
        """Create an instance of the VMPlacementProblemWithIncompFamilies class.

        It simply define the incomp_vm_indexes argument from the config, and then call the parent constructor which will intialize the problem with the basic VM placement constraints.
        """
        config_vmp_with_incomp = config.copy()
        self.families_of_vms : Dict[Any, List[int]] = config_vmp_with_incomp.pop("families_of_vms")
        self.incomp_family_indexes : List[List[Any]]= config_vmp_with_incomp.pop("incomp_family_indexes")
        incomp_vm_indexes: List[List[int]] = []
        for family_indexes in self.incomp_family_indexes:
            for (fam_key_1, fam_key_2) in itertools.combinations(family_indexes, 2):
                for vm_index_1 in self.families_of_vms[fam_key_1]:
                    for vm_index_2 in self.families_of_vms[fam_key_2]:
                        incomp_vm_indexes.append([vm_index_1, vm_index_2])
        config_vmp_with_incomp["incomp_vm_indexes"] = incomp_vm_indexes
        super().__init__(config=config_vmp_with_incomp)
