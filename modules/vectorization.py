from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(student_text_series, project_text_series) -> Tuple[TfidfVectorizer, object, object]:
    vectorizer = TfidfVectorizer()
    corpus = list(student_text_series.astype(str)) + list(project_text_series.astype(str))
    vectorizer.fit(corpus)

    student_matrix = vectorizer.transform(student_text_series.astype(str))
    project_matrix = vectorizer.transform(project_text_series.astype(str))

    return vectorizer, student_matrix, project_matrix
