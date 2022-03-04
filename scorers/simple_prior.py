from abc import abstractmethod

import numpy as np

from .score import Score


class SimplePrior(Score):
    """
        A scorer which does not update its distribution.
        Uses the initial pseudocounts a scoring function for a configuration
    """
    def __init__(self, all_params) -> None:
        max_num_values = len(max(all_params.values(), key=len))
        self.pseudocounts = np.empty((len(all_params), max_num_values))
        self.pseudocounts[:] = np.nan
        super().__init__(all_params)

    def set_pseudocounts(self, pseudocounts):
        for i in range(len(pseudocounts)):
            for j in range(len(pseudocounts[i])):
                self.pseudocounts[i, j] = pseudocounts[i][j]

    def add_positive(self, params):
        return super().add_positive(params)

    def add_negative(self, params):
        return super().add_negative(params)

    @abstractmethod
    def get_score(self, combos):
        pass

    def get_subscore(self, combos):
        return np.zeros(shape=combos.shape[0])


class SortedPrior(SimplePrior):
    """
        Scorer based on the NoUpdatesScorer which computes the best configuration as an argmax of the pseudocounts
    """
    def get_score(self, combos):
        total_pseudo_counts = np.nansum(self.pseudocounts, axis=1)

        # For each config, compute pseudocounts for the used values, for each hyperparameter
        scores = self.pseudocounts[np.arange(combos.shape[1]), combos]

        # Divide those by the total amount of counts+pseudocounts for all values, for each hyperparameter
        scores = scores / total_pseudo_counts

        # For each config, multiply the resulting fractions (of which there are as many as there are hyperparameters)
        scores = np.product(scores, axis=1)
        return scores


class SamplePrior(SimplePrior):
    """
        Scorer based on the NoUpdatesScorer which computes hte best configuration by sampling the pseudocounts
    """
    def get_score(self, combos):
        total_pseudo_counts = np.nansum(self.pseudocounts, axis=1)

        # For each config, compute pseudocounts for the used values, for each hyperparameter
        scores = self.pseudocounts[np.arange(combos.shape[1]), combos]

        # Divide those by the total amount of counts+pseudocounts for all values, for each hyperparameter
        scores = scores / total_pseudo_counts

        # For each config, multiply the resulting fractions (of which there are as many as there are hyperparameters)
        scores = np.product(scores, axis=1)

        # Sample an index using prob distribution of the scores
        idx = np.random.choice(np.arange(combos.shape[0]), p=scores / scores.sum())
        # Make sure the index is chosen by smbo by setting all other indices to 0
        to_return = np.zeros(shape=combos.shape[0])
        to_return[idx] = 1
        return to_return
