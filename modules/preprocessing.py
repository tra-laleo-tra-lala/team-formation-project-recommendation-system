import re

import pandas as pd


SPECIAL_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: str) -> str:
    text = str(value).lower()
    text = text.replace(";", " ").replace(",", " ")
    text = SPECIAL_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_students(df: pd.DataFrame) -> pd.DataFrame:
    students = df.copy()
    students = students.dropna(subset=["student_id", "name", "skills", "interests"]).reset_index(drop=True)

    students["name"] = students["name"].astype(str).str.strip()
    students["skills"] = students["skills"].apply(clean_text)
    students["interests"] = students["interests"].apply(clean_text)

    students["student_profile_text"] = (students["skills"] + " " + students["interests"]).str.strip()
    return students


def preprocess_projects(df: pd.DataFrame) -> pd.DataFrame:
    projects = df.copy()
    projects = projects.dropna(
        subset=["project_id", "project_title", "required_skills", "domain", "description"]
    ).reset_index(drop=True)

    projects["project_title"] = projects["project_title"].astype(str).str.strip()
    projects["required_skills"] = projects["required_skills"].apply(clean_text)
    projects["domain"] = projects["domain"].apply(clean_text)
    projects["description"] = projects["description"].apply(clean_text)

    projects["project_profile_text"] = (
        projects["required_skills"] + " " + projects["description"]
    ).str.strip()
    return projects
