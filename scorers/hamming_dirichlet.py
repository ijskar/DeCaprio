from abc import abstractmethod
import numpy as np

from .score import Score
from parameters import param_order


class HammingDirichlet(Score):
    """
        A scorer based on the Hamming scorer which uses a dirichlet distribution as tie-breaker
    """

    def __init__(self, all_params) -> None:

        max_size = len(max(all_params.values(), key=len))
        self.counts = np.empty(shape=(len(all_params),max_size))
        self.counts[:] = np.nan
        self.pseudocounts = self.counts.copy()

        # Place zeros on the right places
        for i, param in enumerate(param_order):
            self.counts[i, 0:len(all_params[param])] = 0
            self.pseudocounts[i, 0:len(all_params[param])] = 1

        self.latest_improver = np.empty((1, len(all_params)))
        self.latest_improver[:] = np.NAN

        self.one = np.ones((len(all_params)))
        super().__init__(list(all_params.keys()))

    def set_pseudocounts(self, pseudocounts):
        for i in range(len(pseudocounts)):
            for j in range(len(pseudocounts[i])):
                self.pseudocounts[i, j] = pseudocounts[i][j]

    def add_positive(self, params):
        self.latest_improver = params

        mask = np.arange(len(self.counts)), params
        self.counts[mask] += self.one
        return super().add_positive(params)

    @abstractmethod
    def add_negative(self, params):
        pass

    def get_score(self, combos):
        mtrx = np.tile(self.latest_improver, len(combos)).reshape(combos.shape)
        return np.count_nonzero(combos == mtrx, axis=1)

    def get_subscore(self, combos):
        total_counts = np.nansum(self.counts, axis=1)

        total_pseudo_counts = np.nansum(self.pseudocounts, axis=1)

        # For each config, compute counts+pseudocounts for the used values, for each hyperparameter
        subscores = self.counts[np.arange(combos.shape[1]), combos] + \
                 self.pseudocounts[np.arange(combos.shape[1]), combos]

        # Divide those by the total amount of counts+pseudocounts for all values, for each hyperparameter
        subscores = subscores / (total_counts+total_pseudo_counts)

        # For each config, multiply the resulting fractions (of which there are as many as there are hyperparameters)
        subscores = np.product(subscores, axis=1)
        return subscores