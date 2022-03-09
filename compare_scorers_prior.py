import multiprocessing as mp
import pandas as pd
from tqdm.auto import tqdm

import os
from os.path import join, isfile

from compute_pseudocounts import compute_pseudocounts
from scorers.hamming import Hamming
from scorers.beta import Beta
from scorers.dirichlet import Dirichlet
from scorers.hamming_dirichlet import HammingDirichlet
from scorers.baseline import RandomUniform

from algorithms import smbo
from parameters import all_params
from utils import config_id

import warnings

# Supress warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

"""
    Script which compares different scorers with an informed prior for a number of models.
"""


# Directory containing CPMpy models
model_root = "cpmpy_models/"
# Suffix of the models in their saved file (example knapsack.pickle)
model_suffix = ".model"
# Load names of all models
model_names = [fname.replace(model_suffix, "") for fname in os.listdir(model_root)]
# Directory to write results of this experiment
outdir = "scorers_comparison_prior"
# Suffix of the files containing the results of this experiment
out_suffix = "_comparison.pickle"
# Number of runs with different seeds
n_runs = 2

# Directory with results of grid search
grid_search_data = pd.read_pickle("grid_search.pickle")
# Suffix after model name in filenames of precomputed data
#Calculate number of runs in grid search
runs_in_gridsearch = len(grid_search_data.columns.unique(level=1))

def compare_on_model(model_name, verbose=True, model_idx=0):
    """
        This function compares different scorers on a given CPMpy model
    """
    print(f"Comparing search methods on model {model_name}")

    # Load precomputed data
    assert model_name in grid_search_data.columns.levels[0], f"Grid search data for model {model_name} not found"
    precomputed_runtimes = grid_search_data[model_name]
    model=model_name

    # Define scorers to be compared
    scorers = [
               RandomUniform,
               Hamming,
               Beta,
               Dirichlet,
               HammingDirichlet,
            ]

    # Compute priors on all other models
    other_modelnames = [m_name for m_name in model_names if m_name != model_name]
    pseudocounts = compute_pseudocounts(grid_search_data, other_modelnames)
    # Convert pseudocounts to probabilities
    probs = [[i / sum(lst) for i in lst] for lst in pseudocounts]
    # Convert probabilities back to pseudocounts
    pseudocounts = [[i * len(lst) for i in lst] for lst in probs]

    score_names = [scorer.__name__ for scorer in scorers]
    data_entries = ["runtime", "wall", "config_id"]
    columns = pd.MultiIndex.from_product([score_names, range(n_runs), data_entries])
    df = pd.DataFrame(columns=columns)
    for score_class in scorers:
        for seed in range(n_runs):
            #Load data if needed
            use_run = int(seed // (n_runs / runs_in_gridsearch))
            use_run = f"runtime {use_run}"

            # Run SMBO with adaptive capping
            configs = smbo( scorer_class=score_class,
                            m=model,
                            all_params=all_params,
                            verbose=verbose,
                            precomputed_data=precomputed_runtimes,
                            seed=seed,
                            use_column=use_run,
                            pseudocounts=pseudocounts,
                            time_factor=1
                        )

            data = []
            for (time, wall, config) in configs:
                row = {"config_id": config_id(config)}
                row["runtime"] = time
                row['wall'] = wall
                data.append(row)

            df[score_class.__name__, seed] = pd.DataFrame(data, columns=data_entries).astype({"config_id":int, "runtime":float, "wall":float})

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

    print(f"Found {runs_in_gridsearch} runs in gridsearch")

    num_threads = mp.cpu_count()
    pool = mp.Pool(5)

    pool.map(compare_on_model, sorted(model_names))

