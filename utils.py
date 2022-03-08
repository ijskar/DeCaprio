import pandas as pd

from parameters import param_order, all_params
import numpy as np

def config_id(config):
    id = 0
    for i, param in enumerate(param_order):
        param_id = all_params[param].index(config[param])
        if i == len(param_order) - 1:
            factor = 1
        else:
            factor = np.prod([len(all_params[p]) for p in param_order[i + 1:]])
        id += param_id * factor
    return id

def generate_config_ids():

    def all_ids(params):
        if len(params) == 1:
            return {i : tuple([val]) for i, val in enumerate(all_params[params[0]])}

        ids = dict()
        at_idx = 0
        for val in all_params[params[0]]:
            next_ids = all_ids(params[1:])
            ids.update({config_id + at_idx : tuple([val]) + tup for config_id, tup in next_ids.items()})
            at_idx += len(next_ids)
        return ids

    id_dict = all_ids(param_order)
    ids, data = zip(*id_dict.items())
    return pd.DataFrame(data, index=ids, columns=param_order)

if __name__ == "__main__":

    for key, val in generate_config_ids().items():
        print(key, val)

