from typing import Tuple

import pandas as pd


REQUIRED_STUDENT_COLUMNS = ["student_id", "name", "skills", "interests"]
REQUIRED_PROJECT_COLUMNS = ["project_id", "project_title", "required_skills", "domain", "description"]


def _validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_student_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    _validate_columns(df, REQUIRED_STUDENT_COLUMNS)
    return df


def load_project_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    _validate_columns(df, REQUIRED_PROJECT_COLUMNS)
    return df


def load_default_datasets(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    students_path = f"{base_dir}/data/students.csv"
    projects_path = f"{base_dir}/data/projects.csv"

    students_df = load_student_data(students_path)
    projects_df = load_project_data(projects_path)
    return students_df, projects_df
