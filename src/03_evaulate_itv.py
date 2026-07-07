from pathlib import Path
from time import perf_counter
import gc
import pandas as pd
from stanbkt import ColumnNames
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models import MultiBKT

from fit_pybkt_assist import load_pybkt, predict_pybkt
from fit_stanbkt_assist import get_stanbkt_model_path, load_stanbkt
from metrics import compute_metrics

DATA_DIR = Path(__file__).resolve().parents[1] / "data/itv.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output/02_fit"
DEFAULT_STANBKT_MODEL_PATH = OUTPUT_DIR / "itv.stanbktmod"

column_names = {
    ColumnNames.PROBLEM_ID: "problem_id",
    ColumnNames.CORRECTNESS: "correct",
    ColumnNames.STUDENT_ID: "user_id",
    ColumnNames.KC_ID: "skill_name",
    ColumnNames.ORDER: "problem_id",
    ColumnNames.GROUP: "template_id",
}


def evaluate_predictions(label, true_values, pred_values):
    metrics = compute_metrics(true_values, pred_values)
    print(f"{label}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    return metrics


def timed_prediction(predict_fn, model, data, column_mapping, **kwargs):
    start_time = perf_counter()
    true_values, pred_values = predict_fn(
        model, data, column_mapping=column_mapping, **kwargs
    )
    elapsed = perf_counter() - start_time
    return true_values, pred_values, elapsed


def evaluate_with_timing(label, predict_fn, model, train_df, column_mapping, **kwargs):
    train_true, train_pred, train_time = timed_prediction(
        predict_fn, model, train_df, column_mapping=column_mapping, **kwargs
    )
    print(f"{label} Train: {train_time:.4f} seconds")
    print(f"\n{label} Metrics:")
    evaluate_predictions("Train", train_true, train_pred)

    del train_true, train_pred
    gc.collect()


def predict_stanbkt(model: MultiBKT, data, column_mapping, fast=False):
    if not fast:
        predictions = model.predict_posterior_summary(
            data=data, column_mapping=column_mapping, n_cores=1
        )
        print("DONE")

        return predictions["correct"], predictions["pCorrectness_mean"]
    else:
        predictions = model.predict(
            data=data, column_mapping=column_mapping, point_estimate="mean"
        )
        return predictions["correct"], predictions["pCorrectness"]


def main():
    train_df = pd.read_csv(DATA_DIR)

    stanbkt_model = load_stanbkt(load_path=DEFAULT_STANBKT_MODEL_PATH)

    evaluate_with_timing(
        f"StanBKT-MCMC (fast=False)",
        predict_stanbkt,
        stanbkt_model,
        train_df,
        column_names,
        fast=False,
    )
    evaluate_with_timing(
        f"StanBKT-MCMC (fast=True)",
        predict_stanbkt,
        stanbkt_model,
        train_df,
        column_names,
        fast=True,
    )

    del stanbkt_model
    gc.collect()


if __name__ == "__main__":
    main()
