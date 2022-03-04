from cpmpy import *
from cpmpy.solvers.solver_interface import ExitStatus, SolverInterface
from parameters import defaults
from copy import deepcopy
import pandas as pd
import numpy as np


class SavedSolver(SolverInterface):
    """
        Class mimicing a CPMpy solver interface.
        This solver does not actually solve a model. It constructs a SolverStatus object containing the precomputed runtime.
    """
    def __init__(self, solver=None, use_column="mean_runtime"):
        """
            :param solver: Dataframe containing the results of the runs
        """
        assert isinstance(solver, pd.DataFrame)
        super().__init__()

        self.df = solver
        self.use_column = use_column

        n_runs_in_df = len([col for col in self.df.columns if "runtime" in col])
        if use_column == "mean_runtime" and use_column not in self.df.columns:
            self.df['mean_runtime'] = sum(self.df[f'runtime {i}'] for i in range(n_runs_in_df)) / n_runs_in_df

    def solve(self, time_limit=None, **kwargs):
        runtime = self._lookup_runtime(**kwargs)

#       timeout = all(bool(config[f"timeout run {i}"]) for i in range(5))
        timeout = runtime > time_limit

        self.cpm_status.runtime = runtime if not timeout else time_limit
        self.cpm_status.exitstatus = ExitStatus.OPTIMAL if not timeout else ExitStatus.UNKNOWN
        self.cpm_status.solver_name = "DataframeSolver"

        return True

    def _lookup_runtime(self, **kwargs):

        config = deepcopy(kwargs)

        # Fill the missing elements of kwargs with the default values
        for key in defaults:
            if key not in config:
                config[key] = defaults[key]

        condition = np.logical_and.reduce([self.df[column] == value for column, value in config.items()])
        config = self.df[condition].iloc[0]
        return config[self.use_column]

    def status(self):
        return super().status()
