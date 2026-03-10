from typing import Dict, List

import numpy as np
import pandas as pd

from modules.similarity import compute_team_project_similarity


def aggregate_team_vectors(teams: List[List[int]], student_matrix) -> np.ndarray:
    team_vectors = []
    for team in teams:
        team_matrix = student_matrix[team]
        team_vector = np.asarray(team_matrix.mean(axis=0)).ravel()
        team_vectors.append(team_vector)

    if not team_vectors:
        return np.array([])

    return np.vstack(team_vectors)


def recommend_projects_for_teams(
    teams: List[List[int]],
    team_vectors,
    project_df: pd.DataFrame,
    project_matrix,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    if len(teams) == 0:
        return {}

    similarity_matrix = compute_team_project_similarity(team_vectors, project_matrix)
    recommendations = {}

    for team_idx in range(len(teams)):
        sims = similarity_matrix[team_idx]
        top_indices = np.argsort(sims)[::-1][:top_k]

        rec_df = project_df.iloc[top_indices][
            ["project_id", "project_title", "required_skills", "domain", "description"]
        ].copy()
        rec_df["similarity_score"] = sims[top_indices]
        rec_df = rec_df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

        recommendations[f"Team {team_idx + 1}"] = rec_df

    return recommendations
