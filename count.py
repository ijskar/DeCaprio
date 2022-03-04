from score import Score
import numpy as np


class Count(Score):

    def __init__(self, all_params):
        """
            Scorer based on the beta distribution.
        """
        max_size = len(max(all_params.values(), key=len))
        self.counts = np.zeros(shape=(len(all_params), max_size))
        self.success_counts = self.counts.copy()
        self.one = np.ones((len(all_params)))
        super().__init__(list(all_params.keys()))

    def add_positive(self, params):
        mask = np.arange(len(self.counts)), params
        self.counts[mask] += self.one
        self.success_counts[mask] += self.one
        return super().add_positive(params)

    def add_negative(self, params):
        mask = np.arange(len(self.counts)), params
        self.counts[mask] += self.one
        return super().add_negative(params)

    def get_score(self, combos):
        counts = self.counts[np.arange(combos.shape[1]), combos]
        s_counts = self.success_counts[np.arange(combos.shape[1]), combos]

        probs = s_counts / counts

        return np.nanprod(probs, axis=1)

    def get_subscore(self, combos):
        return np.zeros(shape=combos.shape[0])
