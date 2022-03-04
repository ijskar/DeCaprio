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


"""
    Script which compares different scorers for a number of models.
    To speedup this experiment, we are able to use cached runtimes of a grid search.
    Set the following parameters accordingly if you want to use precomputed runtimes.
"""
use_precomputed_data = False

# Directory containing CPMpy models
model_root = "cpmpy_models/"
# Suffix of the models in their saved file (example knapsack.pickle)
model_suffix = ".pickle"
# Directory to write results of this experiment
outdir = "scorers_comparison"
# Suffix of the files containing the results of this experiment
out_suffix = "_comparison.pickle"
# Number of runs with different seeds
n_runs = 10
# Dataframe mapping configurations to ids
config_df = pd.read_pickle("configs.pickle")

## This part is important if you have precomputed data.
## Ignored if use_precomputed_data is False
# Directory with results of grid search
if use_precomputed_data:
    grid_search_dir = "grid_search_results_diff_seeds"
    # Suffix after model name in filenames of precomputed data
    grid_search_suffix = ".pickle"
    #Calculate number of runs in grid search
    dummy = pd.read_pickle(f"{grid_search_dir}/{os.listdir(grid_search_dir)[0]}")
    runs_in_gridsearch = len([col for col in dummy.columns if "runtime" in col])
    print(f"Found {runs_in_gridsearch} runs in gridsearch")


def compare_on_model(model_name, verbose=False):
    print(f"Comparing search methods on model {model_name}")

    # Load precomputed data
    if use_precomputed_data:
        precomputed_runtimes = pd.read_pickle(join(grid_search_dir,f"{model_name}{grid_search_suffix}"))
        model=None
    else:
        precomputed_runtimes = None
        model = Model.from_file(join(model_root, model_name))

    # Define scorers to be compared
    scorers = [RandomUniform,
               Hamming,
               Beta,
               Dirichlet,
               HammingDirichlet,
               ]

    data = []
    for score_class in scorers:
        for seed in range(n_runs):

            if use_precomputed_data:
                use_run = int(seed // (n_runs / runs_in_gridsearch))
                use_run = f"runtime {use_run}"
            else:
                use_run=None

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
                condition = np.logical_and.reduce([config_df[column] == value for column, value in config.items()])
                row = {"config_id": config_df.index[condition].tolist()[0]}
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
    pool = mp.Pool(num_threads - 2)

    model_names = [fname.replace(model_suffix, "") for fname in os.listdir(model_root)]

    pool.map(compare_on_model, sorted(model_names))

