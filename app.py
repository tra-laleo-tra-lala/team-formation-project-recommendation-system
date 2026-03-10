import os
import pandas as pd
import streamlit as st

from modules.data_loader import (
    load_project_data,
    load_student_data,
    load_default_datasets,
)
from modules.preprocessing import preprocess_projects, preprocess_students
from modules.vectorization import build_tfidf_features
from modules.similarity import compute_student_similarity
from modules.team_formation import form_diverse_teams
from modules.recommendation import aggregate_team_vectors, recommend_projects_for_teams
from modules.evaluation import evaluate_recommendations


st.set_page_config(
    page_title="Skill-Based Team Formation and Project Recommendation",
    page_icon="🎯",
    layout="wide",
)


def initialize_state() -> None:
    defaults = {
        "students_raw": None,
        "projects_raw": None,
        "students_df": None,
        "projects_df": None,
        "student_matrix": None,
        "project_matrix": None,
        "vectorizer": None,
        "student_similarity": None,
        "teams": None,
        "team_vectors": None,
        "recommendations": None,
        "metrics_df": None,
        "overall_metrics": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_downstream() -> None:
    st.session_state["teams"] = None
    st.session_state["team_vectors"] = None
    st.session_state["recommendations"] = None
    st.session_state["metrics_df"] = None
    st.session_state["overall_metrics"] = None


def run_preprocessing_and_vectorization() -> None:
    if st.session_state["students_raw"] is None or st.session_state["projects_raw"] is None:
        raise ValueError("Please upload or load both student and project datasets first.")

    students_df = preprocess_students(st.session_state["students_raw"])
    projects_df = preprocess_projects(st.session_state["projects_raw"])

    vectorizer, student_matrix, project_matrix = build_tfidf_features(
        students_df["student_profile_text"],
        projects_df["project_profile_text"],
    )

    st.session_state["students_df"] = students_df
    st.session_state["projects_df"] = projects_df
    st.session_state["vectorizer"] = vectorizer
    st.session_state["student_matrix"] = student_matrix
    st.session_state["project_matrix"] = project_matrix
    st.session_state["student_similarity"] = compute_student_similarity(student_matrix)
    clear_downstream()


def render_home() -> None:
    st.title("Skill-Based Team Formation and Project Recommendation System")
    st.markdown(
        """
        This Streamlit application automatically:

        - Forms balanced student teams using a **greedy diversity algorithm**
        - Uses **TF-IDF vectorization** for profile representation
        - Computes **cosine similarity** for matching
        - Recommends **Top-5 projects** for each team
        - Evaluates recommendations using **Precision@K** and **Skill Coverage**
        """
    )

    st.subheader("Workflow")
    st.write("1. Upload datasets (or load sample data)")
    st.write("2. Preprocess and build TF-IDF features")
    st.write("3. Generate teams")
    st.write("4. Generate project recommendations")
    st.write("5. Evaluate recommendation quality")

    if st.session_state["students_df"] is not None and st.session_state["projects_df"] is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Students", len(st.session_state["students_df"]))
        with col2:
            st.metric("Projects", len(st.session_state["projects_df"]))


def render_upload_page() -> None:
    st.title("Upload Data")

    with st.expander("CSV Format Requirements", expanded=False):
        st.markdown(
            """
            **Student CSV columns:**
            - `student_id`
            - `name`
            - `skills`
            - `interests`

            **Project CSV columns:**
            - `project_id`
            - `project_title`
            - `required_skills`
            - `domain`
            - `description`
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        student_file = st.file_uploader("Upload student CSV", type=["csv"], key="student_upload")
    with col2:
        project_file = st.file_uploader("Upload project CSV", type=["csv"], key="project_upload")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Load Sample Data"):
            students_raw, projects_raw = load_default_datasets(base_dir=os.path.dirname(__file__))
            st.session_state["students_raw"] = students_raw
            st.session_state["projects_raw"] = projects_raw
            st.success("Loaded sample datasets from /data.")

    with c2:
        if st.button("Load Uploaded CSVs"):
            if student_file is None or project_file is None:
                st.error("Please upload both files.")
            else:
                st.session_state["students_raw"] = load_student_data(student_file)
                st.session_state["projects_raw"] = load_project_data(project_file)
                st.success("Uploaded datasets loaded successfully.")

    with c3:
        if st.button("Preprocess + Build Features"):
            try:
                run_preprocessing_and_vectorization()
                st.success("Preprocessing and TF-IDF feature extraction completed.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    if st.session_state["students_raw"] is not None:
        st.subheader("Student Data Preview")
        st.dataframe(st.session_state["students_raw"].head(10), use_container_width=True)

    if st.session_state["projects_raw"] is not None:
        st.subheader("Project Data Preview")
        st.dataframe(st.session_state["projects_raw"].head(10), use_container_width=True)


def render_team_page() -> None:
    st.title("Team Formation")

    if st.session_state["students_df"] is None or st.session_state["student_similarity"] is None:
        st.warning("Please preprocess data first from the Upload Data page.")
        return

    team_size = st.number_input("Team size", min_value=2, max_value=10, value=4, step=1)

    if st.button("Generate Teams"):
        teams = form_diverse_teams(st.session_state["student_similarity"], int(team_size))
        team_vectors = aggregate_team_vectors(teams, st.session_state["student_matrix"])

        st.session_state["teams"] = teams
        st.session_state["team_vectors"] = team_vectors
        st.session_state["recommendations"] = None
        st.session_state["metrics_df"] = None
        st.session_state["overall_metrics"] = None

    if st.session_state["teams"] is None:
        st.info("No teams generated yet.")
        return

    students_df = st.session_state["students_df"]
    sizes = []

    for idx, team in enumerate(st.session_state["teams"], start=1):
        team_df = students_df.iloc[team][["student_id", "name", "skills", "interests"]].reset_index(drop=True)
        sizes.append({"team": f"Team {idx}", "size": len(team)})
        st.subheader(f"Team {idx}")
        st.dataframe(team_df, use_container_width=True)

    st.subheader("Team Size Distribution")
    size_df = pd.DataFrame(sizes).set_index("team")
    st.bar_chart(size_df)


def render_recommendation_page() -> None:
    st.title("Project Recommendation")

    if st.session_state["teams"] is None or st.session_state["team_vectors"] is None:
        st.warning("Please generate teams first.")
        return

    top_k = st.slider("Top-K projects", min_value=1, max_value=10, value=5)

    if st.button("Generate Recommendations"):
        recommendations = recommend_projects_for_teams(
            teams=st.session_state["teams"],
            team_vectors=st.session_state["team_vectors"],
            project_df=st.session_state["projects_df"],
            project_matrix=st.session_state["project_matrix"],
            top_k=top_k,
        )
        st.session_state["recommendations"] = recommendations
        st.session_state["metrics_df"] = None
        st.session_state["overall_metrics"] = None

    if st.session_state["recommendations"] is None:
        st.info("No recommendations generated yet.")
        return

    for team_name, rec_df in st.session_state["recommendations"].items():
        st.subheader(team_name)
        st.dataframe(rec_df, use_container_width=True)

        chart_df = rec_df[["project_title", "similarity_score"]].set_index("project_title")
        st.bar_chart(chart_df)


def render_evaluation_page() -> None:
    st.title("Evaluation")

    if st.session_state["recommendations"] is None:
        st.warning("Please generate recommendations first.")
        return

    default_k = int(next(iter(st.session_state["recommendations"].values())).shape[0])
    k_value = st.number_input("K for Precision@K", min_value=1, max_value=20, value=default_k, step=1)

    if st.button("Run Evaluation"):
        metrics_df, overall_metrics = evaluate_recommendations(
            teams=st.session_state["teams"],
            students_df=st.session_state["students_df"],
            project_df=st.session_state["projects_df"],
            recommendations=st.session_state["recommendations"],
            k=int(k_value),
        )
        st.session_state["metrics_df"] = metrics_df
        st.session_state["overall_metrics"] = overall_metrics

    if st.session_state["metrics_df"] is None:
        st.info("Click 'Run Evaluation' to compute metrics.")
        return

    st.subheader("Per-Team Metrics")
    st.dataframe(st.session_state["metrics_df"], use_container_width=True)

    st.subheader("Overall Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Precision@K", f"{st.session_state['overall_metrics']['avg_precision_at_k']:.3f}")
    with col2:
        st.metric("Average Skill Coverage", f"{st.session_state['overall_metrics']['avg_skill_coverage']:.3f}")


def main() -> None:
    initialize_state()

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Upload Data", "Team Formation", "Project Recommendation", "Evaluation"],
    )

    if page == "Home":
        render_home()
    elif page == "Upload Data":
        render_upload_page()
    elif page == "Team Formation":
        render_team_page()
    elif page == "Project Recommendation":
        render_recommendation_page()
    else:
        render_evaluation_page()


if __name__ == "__main__":
    main()
