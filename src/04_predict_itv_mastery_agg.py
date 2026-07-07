import pandas as pd
import numpy as np
import pathlib
import os
from joblib import Parallel, delayed

DATA_DIR = pathlib.Path(__file__).resolve().parents[1]


def compute_mastery_stats(group_df):
    """Compute mastery stats for a group."""
    pknow = group_df["pKnow"].values
    # Find first occurrence of pKnow > 0.95, else 0
    mastery_idx = np.where(pknow > 0.95)[0]
    mastery_idx = mastery_idx[0] if len(mastery_idx) > 0 else 0
    mastery_max = np.max(pknow)
    return mastery_idx, mastery_max


def process_draw(draw_id, draw_data):
    """Process all groups in a single draw."""
    results = []
    for user_id, group_data in draw_data.groupby("user_id", sort=False):
        mastery_idx, mastery_max = compute_mastery_stats(group_data)
        results.append((draw_id, user_id, mastery_idx, mastery_max))
    return results


# Read data
data = pd.read_csv(DATA_DIR / "output/04_predict/itv_predictions.csv")
print("data loaded")
# Group by draw for processing
grouped_by_draw = list(data.groupby("draw__", sort=False))

# Process all draws in parallel
all_results = Parallel(n_jobs=-1)(
    delayed(process_draw)(draw_id, draw_data) for draw_id, draw_data in grouped_by_draw
)

# Flatten results from all parallel jobs
flat_results = []
for job_result in all_results:
    flat_results.extend(job_result)

# Create output dataframes
mastery_index_df = pd.DataFrame(
    flat_results, columns=["draw__", "user_id", "mastery_index", "_"]
).drop("_", axis=1)

mastery_max_df = pd.DataFrame(
    flat_results, columns=["draw__", "user_id", "_", "mastery_max"]
).drop("_", axis=1)

save_dir = DATA_DIR / "output/04_predict"
os.makedirs(save_dir, exist_ok=True)
mastery_index_df.to_csv(save_dir / "itv_mastery_index.csv", index=False)
mastery_max_df.to_csv(save_dir / "itv_mastery_max.csv", index=False)
