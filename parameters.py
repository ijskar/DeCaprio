# Data file defining the parameters used in the search and their order

all_params = {
    'cp_model_probing_level': [0, 1, 2],
    'preferred_variable_order': [0, 1, 2],
    'linearization_level': [0, 1, 2],
    'symmetry_level': [0, 1, 2],
    'minimization_algorithm': [0, 1, 2],
    'search_branching': [0, 1, 2, 3, 4, 5, 6],
    'optimize_with_core': [False, True],
    'use_erwa_heuristic': [False, True],
    'treat_binary_clauses_separately': [False, True]
}

defaults = {
    'cp_model_probing_level': 2,
    'preferred_variable_order': 0,
    'linearization_level': 1,
    'symmetry_level': 2,
    'minimization_algorithm': 2,
    'search_branching': 0,
    'optimize_with_core': False,
    'use_erwa_heuristic': False,
    'treat_binary_clauses_separately': True
}

param_order = sorted(list(all_params.keys()))

def config_id(config):
    id = 0
    for i, param in enumerate(param_order):
        param_id = all_params[param].index(config[param])
        if i == len(param_order) - 1:
            factor = 1
        else:
            factor = np.prod([len(all_params[p]) for p in param_order[i + 1:]])
        id += param_id * factor
    return id