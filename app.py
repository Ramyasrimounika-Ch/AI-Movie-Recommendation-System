import streamlit as st
import recommender as rec
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Movie Recommender")

st.title("🎬 AI Movie Recommendation System")

st.sidebar.header("Types of Users")

user_type = st.sidebar.radio("User Type", ["New User", "Existing User"])
top_n = st.sidebar.slider("Top N Movies", 5, 35, 10)

genres = ['None','Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller',
           'War', 'Western']

# NEW USER
if user_type == "New User":

    st.subheader("New User Recommendations")

    preferred_genres = st.multiselect(
        "Select preferred genres",
        genres[1:]
    )

    if preferred_genres:
        recs = rec.cold_start_recommendation(preferred_genres, top_n)
        st.subheader(f"Global Top {preferred_genres} Movies")
    else:
        recs = rec.global_top_n(top_n)
        st.subheader("Global Top Movies")

    st.dataframe(recs[["title", "genres", "weighted_rating"]])

    # SHAP explanation (top movie)
    movie = recs.iloc[0]
    st.subheader("Why this movie is recommended?")
    st.info(rec.generate_textual_explanation(None, movie))
    shap_values = rec.explain_movie(movie.movieId)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

else:
    selected_genre = st.sidebar.selectbox("Genre Types", genres)
    user_id = st.sidebar.number_input("User ID", 1, 6040, 1)

    #WHEN NO GENRE IS SELECTED
    if selected_genre == "None":
        st.subheader("Global Top Movies")
        global_movies = rec.global_top_n(top_n)
        st.dataframe(global_movies[["title", "genres", "weighted_rating"]])

        st.subheader("User-Preferred Top Movies")
        personal_movies = rec.personalized_top_n(user_id, top_n)
        st.dataframe(personal_movies[["title", "genres"]])

        top_movie = personal_movies.iloc[0]

        st.subheader("Why this movie?")
        st.info(rec.generate_textual_explanation(user_id, top_movie))

        # Feedback Section
        st.subheader("Give Feedback after watching")

        feedback_source = st.radio(
            "Select list to give feedback on",
            ["Global Top Movies", "User-Preferred Movies"]
        )

        if feedback_source == "Global Top Movies":
            feedback_df = global_movies
        else:
            feedback_df = personal_movies

        feedback_movie = st.selectbox(
            "Select a movie",
            feedback_df["title"]
        )

        movie_id = rec.movies[
            rec.movies["title"] == feedback_movie
        ]["movieId"].values[0]

    # WHEN GENRE IS SELECTED 
    else:
        st.subheader(f"Global Top {selected_genre} Movies")
        global_genre_movies = rec.genre_top_n(selected_genre, top_n)
        st.dataframe(global_genre_movies[["title", "genres", "weighted_rating"]])

        st.subheader(f"User-Preferred {selected_genre} Movies")
        user_genre_movies = rec.user_genre_top_n(user_id, selected_genre, top_n)
        st.dataframe(user_genre_movies[["title", "genres", "weighted_rating"]])

        top_movie = user_genre_movies.iloc[0]

        st.subheader("Why this movie?")
        st.info(rec.generate_textual_explanation(user_id, top_movie))

        # -------- Feedback Section --------
        st.subheader("Give Feedback after watching")

        feedback_source = st.radio(
            "Select list to give feedback on",
            ["Global Genre Movies", "User-Preferred Genre Movies"]
        )

        if feedback_source == "Global Genre Movies":
            feedback_df = global_genre_movies
        else:
            feedback_df = user_genre_movies

        feedback_movie = st.selectbox(
            "Select a movie",
            feedback_df["title"]
        )

        movie_id = rec.movies[
            rec.movies["title"] == feedback_movie
        ]["movieId"].values[0]

    col1, col2 = st.columns(2)

    if col1.button("👍 Like"):
        rec.store_feedback(user_id, movie_id, 1)
        st.success("Feedback saved! Recommendations will improve.")

    if col2.button("👎 Dislike"):
        rec.store_feedback(user_id, movie_id, -1)
        st.warning("Feedback saved! This movie will be deprioritized.")

    st.subheader("Explaining Recommendation")

    shap_values = rec.explain_movie(movie_id)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.subheader("Recommendation Evaluation")

    if st.button("Evaluate Recommendations"):
        precision,recall=rec.precision_recall_at_k(user_id,top_n)

        if precision is not None:
            st.write(f"Precision@{top_n}: {precision:.2f}")
            st.write(f"Recall@{top_n}: {recall:.2f}")
        else:
            st.info("Not enough user history to compute evaluation metrics.")
