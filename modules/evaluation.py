from typing import Dict, List, Set, Tuple

import pandas as pd


def _to_skill_set(text: str) -> Set[str]:
    return {token for token in str(text).split() if token.strip()}


def _team_skill_set(team_indices: List[int], students_df: pd.DataFrame) -> Set[str]:
    skills = set()
    for idx in team_indices:
        skills.update(_to_skill_set(students_df.iloc[idx]["skills"]))
    return skills


def evaluate_recommendations(
    teams: List[List[int]],
    students_df: pd.DataFrame,
    project_df: pd.DataFrame,
    recommendations: Dict[str, pd.DataFrame],
    k: int = 5,
) -> Tuple[pd.DataFrame, dict]:
    rows = []

    for team_num, team in enumerate(teams, start=1):
        team_name = f"Team {team_num}"
        team_skills = _team_skill_set(team, students_df)
        rec_df = recommendations.get(team_name, pd.DataFrame())

        if rec_df.empty:
            rows.append(
                {
                    "team": team_name,
                    "precision_at_k": 0.0,
                    "skill_coverage": 0.0,
                    "relevant_count": 0,
                    "k": k,
                }
            )
            continue

        rec_k = rec_df.head(k)
        relevant_count = 0
        matched_total = 0
        required_total = 0

        for _, project_row in rec_k.iterrows():
            pid = project_row["project_id"]
            original_project = project_df[project_df["project_id"] == pid]
            if original_project.empty:
                continue

            required_skills = _to_skill_set(original_project.iloc[0]["required_skills"])
            matched_skills = team_skills.intersection(required_skills)

            if len(matched_skills) > 0:
                relevant_count += 1

            matched_total += len(matched_skills)
            required_total += len(required_skills)

        precision_at_k = relevant_count / k if k > 0 else 0.0
        skill_coverage = matched_total / required_total if required_total > 0 else 0.0

        rows.append(
            {
                "team": team_name,
                "precision_at_k": precision_at_k,
                "skill_coverage": skill_coverage,
                "relevant_count": relevant_count,
                "k": k,
            }
        )

    metrics_df = pd.DataFrame(rows)

    overall_metrics = {
        "avg_precision_at_k": float(metrics_df["precision_at_k"].mean()) if not metrics_df.empty else 0.0,
        "avg_skill_coverage": float(metrics_df["skill_coverage"].mean()) if not metrics_df.empty else 0.0,
    }

    return metrics_df, overall_metrics
