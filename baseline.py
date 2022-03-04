from score import Score
import numpy as np

class BaselineScore(Score):
    """
        Random search scorer.
        Does not use seen configuraitons
    """

    def __init__(self, all_params) -> None:
        self.latest_improver = None
        super().__init__(all_params)

    def add_positive(self, params):
        self.latest_improver = params
        return super().add_positive(params)

    def add_negative(self, params):
        pass

    def get_score(self, combos):
        return np.zeros(shape=(combos.shape[0]))

    def get_subscore(self, combos):
        return np.zeros(shape=(combos.shape[0]))
