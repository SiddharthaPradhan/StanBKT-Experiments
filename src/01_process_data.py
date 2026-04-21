from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from check_pybkt_backend import require_fast_math


def handle_kc_row(row: object) -> object:
    if pd.isna(row):
        return pd.NA

    row_str = str(row).lstrip("[").rstrip("]")
    kcs_in_row = row_str.split(",")
    if len(kcs_in_row) > 1:
        return pd.NA
    return kcs_in_row[0]


def process_data(
    plogs_path: Path,
    pdets_path: Path,
    top_kcs: int = 10,
    min_problems_per_student_skill: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_fast_math()

    df = pd.read_csv(plogs_path)
    problems_df = pd.read_csv(pdets_path)

    df["start_time"] = pd.to_datetime(
        df["start_time"], format="ISO8601", utc=True, errors="coerce"
    )
    df["start_time_int"] = (
        df["start_time"] - pd.Timestamp("1970-01-01", tz="UTC")
    ) // pd.Timedelta("1ns")

    assert_frame_equal(
        df.sort_values(by=["student_id", "start_time_int"]),
        df.sort_values(by=["student_id", "start_time"]),
    )

    problems_df["skills"] = problems_df["skills"].transform(handle_kc_row)

    df_skills = df.merge(problems_df, on="problem_id", how="inner")
    df_skills.dropna(subset=["skills", "problem_id", "correct"], inplace=True)

    top_kc_names = (
        df_skills.groupby("skills")["correct"]
        .agg("count")
        .sort_values(ascending=False)
        .head(top_kcs)
        .index
    )
    df_skills = df_skills.loc[df_skills["skills"].isin(top_kc_names)].copy()

    df_skills.sort_values(by=["student_id", "start_time_int"], inplace=True)
    df_skills.reset_index(drop=True, inplace=True)

    counts = df_skills.groupby(["student_id", "skills"])["log_id"].transform("count")
    df_skills = df_skills[counts >= min_problems_per_student_skill].copy()
    df_skills.reset_index(drop=True, inplace=True)

    keep_columns = [
        "student_id",
        "problem_id",
        "start_time",
        "start_time_int",
        "correct",
        "skills",
    ]
    df_skills = df_skills.loc[:, keep_columns].copy()

    students = df_skills["student_id"].unique()
    train_students, test_students = train_test_split(
        students, test_size=test_size, random_state=random_state
    )

    train_df = df_skills[df_skills["student_id"].isin(train_students)].copy()
    test_df = df_skills[df_skills["student_id"].isin(test_students)].copy()

    train_kcs = set(train_df["skills"].dropna().unique())
    test_df = test_df[test_df["skills"].isin(train_kcs)].copy()

    unseen_test_kcs = set(test_df["skills"].dropna().unique()) - train_kcs
    assert len(unseen_test_kcs) == 0, f"Unseen KCs in test: {unseen_test_kcs}"

    return df_skills, train_df, test_df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    plogs_path = root / "data" / "plogs.csv"
    pdets_path = root / "data" / "pdets.csv"
    output_dir = root / "output" / "01_data"

    df_skills, train_df, test_df = process_data(
        plogs_path=plogs_path, pdets_path=pdets_path
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Total rows after filtering: {len(df_skills)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved train: {train_path}")
    print(f"Saved test: {test_path}")
    print(
        "Unseen test KCs:",
        len(
            set(test_df["skills"].dropna().unique())
            - set(train_df["skills"].dropna().unique())
        ),
    )


if __name__ == "__main__":
    main()
