import multiprocessing as mp
import pandas as pd
import numpy as np

import os
from os.path import join, isfile

from cpmpy import Model

from scorers.hamming import Hamming
from scorers.beta import Beta
from scorers.dirichlet import Dirichlet
from scorers.hamming_dirichlet import HammingDirichlet
from scorers.baseline import RandomUniform

from algorithms import smbo
from parameters import all_params
from utils import config_id

"""
    Script which compares different scorers for a number of models.
    To speedup this experiment, we are able to use cached runtimes of a grid search.
    Set the following parameters accordingly if you want to use precomputed runtimes.
"""

if __name__ == "__main__":
    use_precomputed_data = True

    # Directory containing CPMpy models
    model_root = "cpmpy_models/"
    # Suffix of the models in their saved file (example knapsack.pickle)
    model_suffix = ".model"
    # Directory to write results of this experiment
    outdir = "scorers_comparison"
    # Suffix of the files containing the results of this experiment
    out_suffix = "_comparison.pickle"
    # Number of runs with different seeds
    n_runs = 10

    ## This part is important if you have precomputed data.
    ## Ignored if use_precomputed_data is False
    # Directory with results of grid search
    if use_precomputed_data:
        grid_search_data = pd.read_pickle("grid_search.pickle")
        # Suffix after model name in filenames of precomputed data
        #Calculate number of runs in grid search
        runs_in_gridsearch = len(grid_search_data.columns.unique(level=1))
        print(f"Found {runs_in_gridsearch} runs in gridsearch")


def compare_on_model(model_name, verbose=False):
    """
        This function compares different scorers on a given CPMpy model
    """
    print(f"Comparing search methods on model {model_name}")

    # Load precomputed data
    if use_precomputed_data:
        assert model_name in grid_search_data.columns.levels[0], f"Grid search data for model {model_name} not found"
        precomputed_runtimes = grid_search_data[model_name]
        model=None
    else:
        precomputed_runtimes = None
        model = Model.from_file(join(model_root, f"{model_name}{model_suffix}"))

    # Define scorers to be compared
    scorers = [
               RandomUniform,
               Hamming,
               Beta,
               Dirichlet,
               HammingDirichlet,
            ]

    data = []
    for score_class in scorers:
        for seed in range(n_runs):
            #Load data if needed
            if use_precomputed_data:
                use_run = int(seed // (n_runs / runs_in_gridsearch))
                use_run = f"runtime {use_run}"
            else:
                use_run=None

            # Run SMBO with adaptive capping
            configs = smbo( scorer_class=score_class,
                            m=model,
                            all_params=all_params,
                            verbose=verbose,
                            precomputed_data=precomputed_runtimes,
                            seed=seed,
                            use_column=use_run,
                            time_factor=1
                        )

            for i, (time, wall, config) in enumerate(configs):
                row = {"config_id": config_id(config)}
                row["runtime"] = time
                row['wall'] = wall
                row['seed'] = seed
                row["iteration"] = i
                row["algorithm"] = score_class.__name__
                data.append(row)

    columns = ["algorithm", "seed", "iteration","runtime", "wall", "config_id"]
    df = pd.DataFrame(data, columns = columns)

    filename_runtimes = join(outdir, f"{model_name}{out_suffix}")

    print(f"Writing dataframe to {filename_runtimes}")
    df.to_pickle(filename_runtimes)


if __name__ == "__main__":

    assert outdir is not None, f"outdir cannot be None"

    try:
        os.mkdir(outdir)
    except FileExistsError as e:
        print(f"Warning: directory with results already exists, containing {len(os.listdir(outdir))} files")
        if input("Press y to overwrite results ") != 'y':
            exit(1)
        pass

    num_threads = mp.cpu_count()
    pool = mp.Pool(1)

    model_names = [fname.replace(model_suffix, "") for fname in os.listdir(model_root)]
    pool.map(compare_on_model, sorted(model_names))

