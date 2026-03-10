from typing import List

import numpy as np


def _average_similarity_to_team(candidate: int, team: list, similarity_matrix: np.ndarray) -> float:
    if not team:
        return 0.0
    return float(np.mean([similarity_matrix[candidate, member] for member in team]))


def form_diverse_teams(similarity_matrix: np.ndarray, team_size: int = 4) -> List[List[int]]:
    if team_size < 2:
        raise ValueError("team_size must be at least 2")

    n_students = similarity_matrix.shape[0]
    if n_students == 0:
        return []

    unassigned = set(range(n_students))
    teams = []

    while unassigned:
        starter = min(
            unassigned,
            key=lambda idx: float(
                np.mean(
                    [similarity_matrix[idx, j] for j in unassigned if j != idx]
                )
            )
            if len(unassigned) > 1
            else 0.0,
        )

        team = [starter]
        unassigned.remove(starter)

        while len(team) < team_size and unassigned:
            next_member = min(
                unassigned,
                key=lambda candidate: _average_similarity_to_team(candidate, team, similarity_matrix),
            )
            team.append(next_member)
            unassigned.remove(next_member)

        teams.append(team)

    return teams
