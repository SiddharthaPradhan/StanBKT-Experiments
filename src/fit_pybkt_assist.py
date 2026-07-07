from pyBKT.models import Model
import pandas as pd
from time import perf_counter
import os
from pathlib import Path
from check_pybkt_backend import require_fast_math


DEFAULT_PYBKT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "output/02_fit/pybkt_model.pkl"
)


def fit_pybkt(data, seed, save_path=DEFAULT_PYBKT_MODEL_PATH):
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
    require_fast_math()

    model = Model(seed=seed)

    defaults = {
        "order_id": "start_time_int",
        "skill_name": "skills",
        "user_id": "student_id",
    }
    start_time = perf_counter()
    # Fit the model using the provided data and additional kwargs
    model.fit(data=data, forgets=True, defaults=defaults)
    print("TIME:")
    print("Fitting pyBKT model took {:.2f} seconds".format(perf_counter() - start_time))
    # Save the fitted model to the specified path
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    model.save(save_path)

    return model


def predict_pybkt(model, data):
    predictions = model.predict(data=data)
    return predictions["correct"], predictions["correct_predictions"]


def load_pybkt(load_path=DEFAULT_PYBKT_MODEL_PATH):
    require_fast_math()

    model = Model()
    model.load(Path(load_path))
    return model
