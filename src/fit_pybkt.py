from pyBKT.models import Model
import pandas as pd
from time import perf_counter

def fit_pybkt(data, seed, save_path="./output/02_fit/pybkt_model.pkl", **kwargs):
    """
    Fit the and save it pyBKT model to the given data.

    Parameters
    ----------
    data : pd.DataFrame
        The input data in long format, with columns 'user_id', 'item_id', 'correct', and optionally 'skill_id'.
    seed : int
        The random seed for reproducibility.
    save_path : str, optional
        The path to save the fitted pyBKT model. Default is "./output/02_fit/pybkt_model.pkl".

    Returns
    -------
    None
    """
    model = Model(seed=seed)

    defaults = {'order_id': 'start_time_int', 'skill_name': 'skills', }
    # Fit the model using the provided data and additional kwargs
    model.fit(data=data, forgets=True)

    # Save the fitted model to the specified path
    model.save(save_path)

    model.predict(data=data)

    return model