from pathlib import Path
from time import perf_counter

from stanbkt import ColumnNames
from stanbkt import MCMCFitOptions, MLEFitOptions, PFFitOptions, VBFitOptions
from stanbkt import load_model
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models import StandardBKT

# mapping
column_names = {
    ColumnNames.PROBLEM_ID: "problem_id",
    ColumnNames.CORRECTNESS: "correct",
    ColumnNames.STUDENT_ID: "student_id",
    ColumnNames.KC_ID: "skills",
    ColumnNames.ORDER: "start_time_int",
}
N_PARALLEL_CHAINS = 4
N_THREADS_PER_CHAIN = 8
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output/02_fit"
DEFAULT_STANBKT_MODEL_PATH = OUTPUT_DIR / "stanbkt.stanbktmod"
DEFAULT_STANBKT_VB_MODEL_PATH = OUTPUT_DIR / "stanbkt_vb.stanbktmod"
DEFAULT_STANBKT_MLE_MODEL_PATH = OUTPUT_DIR / "stanbkt_mle.stanbktmod"
DEFAULT_STANBKT_PATHFINDER_MODEL_PATH = OUTPUT_DIR / "stanbkt_pathfinder.stanbktmod"

DEFAULT_STANBKT_MODEL_PATHS = {
    FitMethod.MCMC: DEFAULT_STANBKT_MODEL_PATH,
    FitMethod.VB: DEFAULT_STANBKT_VB_MODEL_PATH,
    FitMethod.MLE: DEFAULT_STANBKT_MLE_MODEL_PATH,
    FitMethod.PATHFINDER: DEFAULT_STANBKT_PATHFINDER_MODEL_PATH,
}


def resolve_fit_method(fit_method: FitMethod | str) -> FitMethod:
    return FitMethod(fit_method)


def get_stanbkt_model_path(fit_method: FitMethod | str) -> Path:
    return DEFAULT_STANBKT_MODEL_PATHS[resolve_fit_method(fit_method)]


def build_stan_fit_options(fit_method: FitMethod | str, seed: int) -> StanFitOptions:
    resolved_fit_method = resolve_fit_method(fit_method)
    if resolved_fit_method == FitMethod.MCMC:
        return MCMCFitOptions(
            seed=seed,
            show_console=True,
            show_progress=False,
            threads_per_chain=N_THREADS_PER_CHAIN,
            chains=N_PARALLEL_CHAINS,
            parallel_chains=N_PARALLEL_CHAINS,
        )
    if resolved_fit_method == FitMethod.VB:
        return VBFitOptions(seed=seed)
    if resolved_fit_method == FitMethod.MLE:
        return MLEFitOptions(seed=seed)
    if resolved_fit_method == FitMethod.PATHFINDER:
        return PFFitOptions(
            seed=seed, num_threads=N_THREADS_PER_CHAIN, show_console=True
        )
    raise ValueError(f"Unsupported StanBKT fit method: {resolved_fit_method}")


def fit_stanbkt(
    data,
    seed,
    save_path=None,
    fit_method: FitMethod | str = FitMethod.MCMC,
):
    """
    Fit and save a StanBKT model with the requested Stan inference method.

    Parameters
    ----------
    data : pd.DataFrame
        The input data in long format, with columns 'user_id', 'item_id', 'correct', and optionally 'skill_id'.
    seed : int
        The random seed for reproducibility.
    save_path : str or Path, optional
        The path to save the fitted StanBKT model. Defaults to a method-specific
        artifact path under output/02_fit.
    fit_method : FitMethod or str, default FitMethod.MCMC
        Stan inference method. Supported values are "mcmc", "vb", "mle",
        and "pathfinder".

    Returns
    -------
    StandardBKT
        The fitted StanBKT model.
    """
    resolved_fit_method = resolve_fit_method(fit_method)
    resolved_save_path = (
        Path(save_path)
        if save_path is not None
        else get_stanbkt_model_path(resolved_fit_method)
    )

    cpp_opts = {"STAN_THREADS": True}

    model = StandardBKT(fit_method=resolved_fit_method, cpp_compile_kwargs=cpp_opts)

    stan_fit_opts = build_stan_fit_options(resolved_fit_method, seed)
    print(f"StanBKT fit method: {resolved_fit_method.value}")
    print(stan_fit_opts)
    start_time = perf_counter()
    # Fit the model using the provided data and additional kwargs
    model.fit(data=data, column_mapping=column_names, stan_fit_options=stan_fit_opts)
    print("TIME:")
    print(
        f"Fitting {resolved_fit_method.value} took {perf_counter() - start_time} seconds"
    )
    # Save the fitted model to the specified path
    model.save(resolved_save_path)

    return model


def predict_stanbkt(model: StandardBKT, data, fast=False):
    if not fast:
        predictions = model.predict_posterior_summary(
            data=data, column_mapping=column_names, n_cores=1
        )
        print("DONE")

        return predictions["correct"], predictions["pCorrectness_mean"]
    else:
        predictions = model.predict(
            data=data, column_mapping=column_names, point_estimate="mean"
        )
        return predictions["correct"], predictions["pCorrectness"]


def load_stanbkt(load_path=DEFAULT_STANBKT_MODEL_PATH):
    return load_model(Path(load_path))
