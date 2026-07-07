from stanbkt import load_model
from stanbkt.models import MultiBKT
from typing import cast

import pandas as pd
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parents[1]

data = pd.read_csv(DATA_DIR / "data/itv.csv")
itv_model: MultiBKT = cast(
    MultiBKT, load_model(DATA_DIR / "output/02_fit/itv.stanbktmod")
)
from stanbkt import ColumnNames

# extract the correct stan index to group from the data
# in newer stanbkt, this is stored in the fit object
column_names = {
    ColumnNames.PROBLEM_ID: "problem_id",
    ColumnNames.CORRECTNESS: "correct",
    ColumnNames.STUDENT_ID: "user_id",
    ColumnNames.KC_ID: "skill_name",
    ColumnNames.ORDER: "problem_id",
    ColumnNames.GROUP: "template_id",
}

preds = itv_model.predict_smoothed_posterior_draws(data, column_names)[
    "order-of-operations rule"
]
import os

save_dir = DATA_DIR / "output/04_predict"
os.makedirs(save_dir, exist_ok=True)
preds.to_csv(save_dir / "itv_predictions.csv", index=False)
