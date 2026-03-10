import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_student_similarity(student_matrix) -> np.ndarray:
    return cosine_similarity(student_matrix)


def compute_team_project_similarity(team_vectors, project_matrix) -> np.ndarray:
    return cosine_similarity(team_vectors, project_matrix)
