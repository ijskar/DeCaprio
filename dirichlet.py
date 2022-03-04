from score import Score
from math import gamma
from abc import abstractmethod
import numpy as np
from parameters import param_order


class DirichletScore(Score):
    """
        Scorer based on the Dirichlet distribution
    """
    def __init__(self, all_params) -> None:
        max_num_values = len(max(all_params.values(), key=len))
        # Initialize arrays for counts and pseudocounts
        self.counts = np.empty((len(all_params), max_num_values))
        self.counts[:] = np.nan
        self.pseudocounts = self.counts.copy()

        # Place zeros on the right place
        for i, param in enumerate(param_order):
            self.counts[i, 0:len(all_params[param])] = 0
            self.pseudocounts[i, 0:len(all_params[param])] = 1

        self.one = np.ones((len(all_params)))
        super().__init__(all_params)

    def set_pseudocounts(self, pseudocounts):
        # Initialize the distribution
        for i in range(len(pseudocounts)):
            for j in range(len(pseudocounts[i])):
                self.pseudocounts[i, j] = pseudocounts[i][j]

    def add_positive(self, params):
        # Increment the counts corresponding to the values in params with 1
        mask = np.arange(len(self.counts)), params
        self.counts[mask] += self.one
        return super().add_positive(params)

    def add_negative(self, params, mu=1):
        # Add count of mu to all other values
        for i in range(len(params)):
            self.counts[i, np.arange(self.counts.shape[1]) != params[i]] += mu
        return super().add_negative(params)

    def get_score(self, combos):
        total_counts = np.nansum(self.counts, axis=1)

        total_pseudo_counts = np.nansum(self.pseudocounts, axis=1)

        # For each config, compute counts+pseudocounts for the used values, for each hyperparameter
        scores = self.counts[np.arange(combos.shape[1]), combos] +\
                 self.pseudocounts[np.arange(combos.shape[1]), combos]

        # Divide those by the total amount of counts+pseudocounts for all values, for each hyperparameter
        scores = scores / (total_counts+total_pseudo_counts)

        # For each config, multiply the resulting fractions (of which there are as many as there are hyperparameters)
        scores = np.product(scores, axis=1)
        return scores

    def get_subscore(self, combos):
        return np.zeros(shape=combos.shape[0])

