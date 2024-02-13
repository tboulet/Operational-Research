import pyomo.environ as pyo

variable_string_type_to_pyomo_domain = {
    "continuous": pyo.Reals,
    "integer": pyo.Integers,
    "binary": pyo.Binary,
}
variable_string_sense_to_pyomo_sense = {
    "minimize": 1,
    "maximize": -1,
}