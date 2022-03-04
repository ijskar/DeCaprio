import time

from tqdm import tqdm
from cpmpy.solvers import CPM_ortools, param_combinations

from saved_solver import SavedSolver
import random
from cpmpy.solvers.solver_interface import ExitStatus
import numpy as np
from parameters import param_order, defaults


def params_to_np(combos):
    """
        :param combos: a dictionary with (parameter_name, value) pairs
        :return: a numpy array with the values of the parameters in predefined order
    """
    arr = [[params[key] for key in param_order] for params in combos]
    return np.array(arr)


def np_to_params(arr):
    """
        :param arr: a numpy array with the values of the parameters in predefined order
        :return: a dictionary with (parameter_name, value) pairs
    """
    return {key : val for key, val in zip(param_order, arr)}


def basic_grid_search(m, all_params, verbose=False, seed=0):
    """
        Runs a full random grid search on the parameter search space"
        :param m: CPMpy model to evaluate parameter configurations on
        :param all_params: dictionary with (parameter_name, [values]) pairs
        :return: the ordered list of tested configurations with their runtime
    """
    configs = []  # (runtime, param)
    config_list = list(param_combinations(all_params))
    random.shuffle(config_list)
    if verbose:
        pbar = tqdm(config_list)
    for params in config_list:
        s = CPM_ortools(m)
        s.solve(num_search_workers=1, random_seed=seed, **params)
        configs.append((s.status().runtime, params))
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()
    return configs


def timeout_grid_search(m, all_params, verbose=False, seed=0):
    """
        Runs a full grid search with adaptive capping
        :param m: the CPMpy model to evaluate parameter configurations on
        :param all_params: dictionary with (parameter_name, [values]) pairs
        :return: the ordered list of tested configurations with their (capped) runtime
    """
    # Determine base runtime
    m.solve()
    base_runtime = m.status().runtime

    configs = []  # (runtime, param)
    config_list = list(param_combinations(all_params))
    random.shuffle(config_list)
    if verbose:
        pbar = tqdm(config_list)
    for params in config_list:
        s = CPM_ortools(m)
        # run configuration
        s.solve(num_search_workers=1, 
                random_seed=seed, 
                **params, 
                time_limit=1.05 * base_runtime)  # timeout of 105% of base_runtime

        configs.append((s.status().runtime, params))
        if s.status().exitstatus == ExitStatus.OPTIMAL and s.status().runtime < base_runtime:
            base_runtime = s.status().runtime
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()
    return configs


def smbo(scorer_class, m, all_params, verbose=True, time_factor=1.05, seed=0, precomputed_data=None, pseudocounts=None, use_column=None):
    """
        Runs DeCaprio version of SMBO algorithm
        :param scorer_class: subclass of Scorer used to rank remaining configurations
        :param m: CPMpy model to evaluate parameter configurations on
        :param all_params: dictionary with (parameter_name, [values]) pairs
        :param time_factor: threshold for timing out the next configuration compared to the current best
        :param seed: random seed
        :param precomputed_data: dataframe with runtime results of all configuration on this model.
                                 this allows the algorithm to lookup the runtime instead of running the actual solver
        :param pseudocounts: pseudocounts of parameter values to intialize scoring functions which support it
        :param use_column: determines which column of the results in the precomputed data to use
        :return: the ordered list of tested configurations with their (capped) runtime
    """
    
    if precomputed_data is not None:
        assert use_column is not None, "'use_column' cannot be None when using precomputed data"

    if precomputed_data is None:
        assert m is not None, "model cannot be None if no precomputed data is provided"

    configs = []

    # Determine initial best runtime
    if precomputed_data is not None:
        # Use a fake solver looking up the runtimes
        s = SavedSolver(solver=precomputed_data, use_column=use_column)
        best_runtime = s._lookup_runtime(**defaults)

    else:
        # Use an actual solver
        s = CPM_ortools(m)
        s.solve(num_search_workers=1, random_seed=seed)
        best_runtime = s.status().runtime

    # Add default's runtime as first entry in configs
    configs.append((best_runtime, 0, defaults))

    combos = list(param_combinations(all_params))

    scorer = scorer_class(all_params)
    if pseudocounts is not None:
        scorer.set_pseudocounts(pseudocounts)

    combos_np = params_to_np(combos)

    np.random.seed(seed)
    # Ensure random start
    np.random.shuffle(combos_np)

    if verbose:
        pbar = tqdm(total=len(combos))

    while len(combos_np):
        start = time.time()
        # Apply scoring to all combos
        scores = scorer.get_score(combos_np)
        max_idx = np.where(scores == scores.max())[0]
        # Tie breaking using sub_score
        sub_scores = scorer.get_subscore(combos_np[max_idx])
        # Get index of optimal combo
        param_idx = max_idx[np.argmax(sub_scores)]
        params_np = combos_np[param_idx]
        # Remove optimal combo from combos
        combos_np = np.delete(combos_np, param_idx, axis=0)
        # Convert numpy array back to dictionary
        params_dict = np_to_params(params_np)
        
        # Evaluate chosen configuration
        if precomputed_data is not None:
            s.solve(time_limit=time_factor * best_runtime, **params_dict)
        else:
            s.solve(num_search_workers=1, random_seed=seed, **params_dict, time_limit=time_factor * best_runtime)

        if s.status().exitstatus == ExitStatus.OPTIMAL and s.status().runtime < best_runtime:
            best_runtime = s.status().runtime
            # update surrogate
            scorer.add_positive(params_np)
        else:
            # update surrogate
            scorer.add_negative(params_np)

        wall = time.time() - start
        configs.append((s.status().runtime, wall, params_dict))
        if verbose:
            pbar.update()

    if verbose:
        pbar.close()

    return configs
