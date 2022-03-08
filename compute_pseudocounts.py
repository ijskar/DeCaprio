import utils
from parameters import all_params, param_order
import pandas as pd
import os

"""
    Script to compute the pseudocounts for the Dirichlet distributions using the results of a full grid search.
    Pseudocounts are calculated by defining a gap to optimality in terms of runtime.
    All configurations which fall in this band are considered. 
    For each parameter value, we count the number of occurences in this pool of configurations
"""

def compute_pseudocounts(grid_search_data, model_names):
    """
    Computes pseudocounts for dirichlet distributions
    :param grid_search_data: A :pandas.DataFrame: containing the results of a full grid search
    :param result_file_names: A list of model names to use in the computation of the pseudocounts
    :return: A dictionary that maps hyperparameter names onto lists of associated pseudocounts
    """
    pseudocounts = [[0]*len(all_params[param]) for param in param_order]

    model_data = grid_search_data[model_names]
    avg_model_data = model_data.groupby(axis="columns", level=0).mean()
    # Find the minimum runtime for each model
    min_runtimes = avg_model_data.min(axis="rows")

    #Find all configs in bounds of best configs
    best_configs = []
    for name in model_names:
        ids = avg_model_data[avg_model_data[name] <= 1.05 * min_runtimes[name]].index
        best_configs += list(ids)

    configs = utils.generate_config_ids()

    best = configs.loc[best_configs]

    for i, param in enumerate(param_order):
        counts = best[param].value_counts()
        for val, count in counts.iteritems():
            idx = all_params[param].index(val)
            pseudocounts[i][idx] = count

        assert sum(pseudocounts[i]) == len(best_configs), "An error occured, pseudocounts should sum to number of best configs"

    return pseudocounts

if __name__ == "__main__":
    grid_search_data = pd.read_pickle("grid_search.pickle") # Change this line to load different grid search data
    model_names = ["3sum", "among"]