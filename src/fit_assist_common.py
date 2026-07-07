from pathlib import Path

import pandas as pd

from fit_pybkt_assist import DEFAULT_PYBKT_MODEL_PATH, fit_pybkt
from fit_stanbkt_assist import fit_stanbkt, get_stanbkt_model_path

DATA_DIR = Path(__file__).resolve().parents[1] / "output/01_data"
DEFAULT_SEED = 12345


def run_fit_assist(fit_method: str, include_pybkt: bool = True) -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    if include_pybkt:
        fit_pybkt(train_df, seed=DEFAULT_SEED, save_path=DEFAULT_PYBKT_MODEL_PATH)

    stanbkt_path = get_stanbkt_model_path(fit_method)
    fit_stanbkt(
        train_df, seed=DEFAULT_SEED, save_path=stanbkt_path, fit_method=fit_method
    )

    print("Saved fitted models:")
    if include_pybkt:
        print(f"pyBKT: {DEFAULT_PYBKT_MODEL_PATH}")
    print(f"StanBKT ({fit_method}): {stanbkt_path}")
