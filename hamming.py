import numpy as np

from score import Score


class HammingScore(Score):
    """
        Scorer based on the Hamming distance between compared to the current best configuration
    """
    def __init__(self, all_params) -> None:
        self.latest_improver = np.empty((1, len(all_params)))
        self.latest_improver[:] = np.nan
        super().__init__(all_params)

    def add_positive(self, params):
        self.latest_improver = params

        return super().add_positive(params)
        
    def add_negative(self, params):
        pass

    def get_score(self, combos):
        mtrx = np.tile(self.latest_improver,len(combos)).reshape(combos.shape)
        return np.count_nonzero(combos == mtrx, axis=1)

    def get_subscore(self, combos):
        return np.zeros(shape=(combos.shape[0]))


