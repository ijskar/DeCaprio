

from abc import abstractmethod


class Score(object):

    def __init__(self, all_params) -> None:
        super().__init__()
        self.headers = [x for x in all_params]
        self.history = []

    def set_pseudocounts(self, pseudocounts):
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return type(self).__name__

    @abstractmethod
    def add_positive(self, params):
        """
            Add a positive configuration to the scorer.
            A configuration is positive if it improved the current best configuration
        """
        self.history.append(params)

    @abstractmethod
    def add_negative(self, params):
        """
            Add a negative configuration to the scorer
            A configuration is negative it does not improve on the current best configuration
        """
        pass

    def get_score(self, params):
        """
            Returns the score for an array of configurations
            :param params: An numpy matrix representing configurations
                           A row for every configuration
                           A column of every parameter as defined in parameters.py
        """
        raise NotImplementedError("get_score is not implemented")

    def get_subscore(self,params):
        """
            Returns the subscore for a configuration
            :param params: An numpy matrix representing configurations
                           A row for every configuration
                           A column of every parameter as defined in parameters.py
        """
        raise NotImplementedError("get_subscore is not implemented")
