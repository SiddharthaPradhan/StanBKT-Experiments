from pathlib import Path
from time import perf_counter
import gc
import pandas as pd
from stanbkt.fits.fit_types import FitMethod

from fit_pybkt_assist import load_pybkt, predict_pybkt
from fit_stanbkt_assist import get_stanbkt_model_path, load_stanbkt, predict_stanbkt
from metrics import compute_metrics

DATA_DIR = Path(__file__).resolve().parents[1] / "output/01_data"


def evaluate_predictions(label, true_values, pred_values):
    metrics = compute_metrics(true_values, pred_values)
    print(f"{label}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    return metrics


def timed_prediction(predict_fn, model, data, **kwargs):
    start_time = perf_counter()
    true_values, pred_values = predict_fn(model, data, **kwargs)
    elapsed = perf_counter() - start_time
    return true_values, pred_values, elapsed


def evaluate_with_timing(label, predict_fn, model, train_df, test_df, **kwargs):
    train_true, train_pred, train_time = timed_prediction(
        predict_fn, model, train_df, **kwargs
    )
    test_true, test_pred, test_time = timed_prediction(
        predict_fn, model, test_df, **kwargs
    )

    print(f"\n{label} Metrics:")
    evaluate_predictions("Train", train_true, train_pred)
    evaluate_predictions("Test", test_true, test_pred)
    print(f"{label} Train: {train_time:.4f} seconds")
    print(f"{label} Test: {test_time:.4f} seconds")

    del train_true, train_pred, test_true, test_pred
    gc.collect()


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    pybkt_model = load_pybkt()
    evaluate_with_timing("pyBKT", predict_pybkt, pybkt_model, train_df, test_df)

    del pybkt_model
    gc.collect()

    for fit_method in [
        # FitMethod.VB,
        # FitMethod.MLE,
        FitMethod.MCMC,
        FitMethod.PATHFINDER,
    ]:
        model_path = get_stanbkt_model_path(fit_method)
        if not model_path.exists():
            print(
                f"\nSkipping StanBKT-{fit_method.value.upper()}: model not found at {model_path}"
            )
            continue

        stanbkt_model = load_stanbkt(load_path=model_path)
        if not fit_method == FitMethod.PATHFINDER:
            evaluate_with_timing(
                f"StanBKT-{fit_method.value.upper()} (fast=False)",
                predict_stanbkt,
                stanbkt_model,
                train_df,
                test_df,
                fast=False,
            )
        evaluate_with_timing(
            f"StanBKT-{fit_method.value.upper()} (fast=True)",
            predict_stanbkt,
            stanbkt_model,
            train_df,
            test_df,
            fast=True,
        )

        del stanbkt_model
        gc.collect()


if __name__ == "__main__":
    main()
