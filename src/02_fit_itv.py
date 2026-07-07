from pathlib import Path
import pandas as pd
from stanbkt import ColumnNames, MCMCFitOptions, MCMCFitOptions, StandardBKT, FitMethod
from stanbkt.models import MultiBKT

data_file = Path(__file__).resolve().parents[1] / "data/itv.csv"
# mapping
column_names = {
    ColumnNames.PROBLEM_ID: "problem_id",
    ColumnNames.CORRECTNESS: "correct",
    ColumnNames.STUDENT_ID: "user_id",
    ColumnNames.KC_ID: "skill_name",
    ColumnNames.ORDER: "problem_id",
    ColumnNames.GROUP: "template_id",
}

N_PARALLEL_CHAINS = 8
N_THREADS_PER_CHAIN = 8
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output/02_fit"
DEFAULT_STANBKT_MODEL_PATH = OUTPUT_DIR / "itv.stanbktmod"


def main():
    df = pd.read_csv(data_file)
    model = MultiBKT(
        fit_method=FitMethod.MCMC, cpp_compile_kwargs={"STAN_THREADS": True}
    )
    opts = MCMCFitOptions(
        seed=12345,
        show_console=True,
        show_progress=False,
        threads_per_chain=N_THREADS_PER_CHAIN,
        chains=N_PARALLEL_CHAINS,
        parallel_chains=N_PARALLEL_CHAINS,
    )
    model.fit(
        df,
        column_mapping=column_names,
        stan_fit_options=opts,
    )
    model.save(DEFAULT_STANBKT_MODEL_PATH)


if __name__ == "__main__":
    main()
