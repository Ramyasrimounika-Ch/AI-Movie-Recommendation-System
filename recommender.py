import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import shap

movies = pd.read_csv("ml-1m/movies.dat",sep="::",engine="python",names=["movieId", "title", "genres"],encoding="latin-1")
ratings = pd.read_csv("ml-1m/ratings.dat",sep="::",engine="python",names=["userId", "movieId", "rating", "timestamp"])

data = ratings.merge(movies, on="movieId")

movie_stats = data.groupby("movieId").agg(
    v=("rating", "count"),
    R=("rating", "mean")
).reset_index()

C = data["rating"].mean()
m = movie_stats["v"].quantile(0.90)

def imdb_weighted_rating(row):
    return (row.v / (row.v + m)) * row.R + (m / (row.v + m)) * C

movie_stats["weighted_rating"] = movie_stats.apply(imdb_weighted_rating, axis=1)
movie_scores = movie_stats.merge(movies, on="movieId")
weighted_ratings = movie_stats[["movieId", "weighted_rating"]]

user_movie_matrix = data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_movie_filled = user_movie_matrix.fillna(0)

user_similarity = cosine_similarity(user_movie_filled)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_filled.index,
    columns=user_movie_filled.index
)

user_feedback = {}

def store_feedback(user_id, movie_id, feedback):
    if user_id not in user_feedback:
        user_feedback[user_id] = {}
    user_feedback[user_id][movie_id] = feedback

def global_top_n(top_n=10):
    return movie_scores.sort_values(
        "weighted_rating", ascending=False
    ).head(top_n)

def genre_top_n(genre, top_n=10):
    return movie_scores[
        movie_scores["genres"].str.contains(genre, case=False)
    ].sort_values("weighted_rating", ascending=False).head(top_n)

def cold_start_recommendation(preferred_genres, top_n=10):
    pattern = "|".join(preferred_genres)
    return movie_scores[
        movie_scores["genres"].str.contains(pattern, case=False)
    ].sort_values("weighted_rating", ascending=False).head(top_n)

def personalized_top_n(user_id, top_n=10, remove_watched=True):
    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1:6]
    )

    scores = np.zeros(user_movie_filled.shape[1])

    for sim_user, sim in similar_users.items():
        scores += sim * user_movie_filled.loc[sim_user].values

    scores = pd.Series(scores, index=user_movie_filled.columns)

    if user_id in user_feedback:
        for movie_id, fb in user_feedback[user_id].items():
            if movie_id in scores:
                scores[movie_id] += fb

    if remove_watched:
        watched = user_movie_matrix.loc[user_id].dropna().index
        scores = scores.drop(watched, errors="ignore")

    return (
        scores.sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .merge(movies, on="movieId")
    )

def user_genre_top_n(user_id, genre, top_n=10):
    user_movies = ratings[ratings["userId"] == user_id]["movieId"]

    genre_movies = movies[
        movies["genres"].str.contains(genre, case=False, na=False)
    ]
    genre_movies = genre_movies[
        ~genre_movies["movieId"].isin(user_movies)
    ]
    genre_movies = genre_movies.merge(
        weighted_ratings,
        on="movieId",
        how="left"
    )
    return genre_movies.sort_values(
        "weighted_rating", ascending=False
    ).head(top_n)

#SHAP Explainability
X = movie_stats[["v", "R"]]
y = movie_stats["weighted_rating"]

explain_model = LinearRegression()
explain_model.fit(X, y)

shap_explainer = shap.Explainer(explain_model, X)

def explain_movie(movie_id):
    row = movie_stats[movie_stats["movieId"] == movie_id][["v", "R"]]
    return shap_explainer(row)

def get_user_genre_profile(user_id, top_k=3):
    """
    Builds a user genre preference profile based on past ratings.
    Returns top-k preferred genres.
    """
    user_history = data[data["userId"] == user_id]

    if user_history.empty:
        return []

    genre_scores = {}

    for _, row in user_history.iterrows():
        rating = row["rating"]
        genres = row["genres"].split("|")

        for g in genres:
            genre_scores[g] = genre_scores.get(g, 0) + rating

    sorted_genres = sorted(
        genre_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [g for g, _ in sorted_genres[:top_k]]

def generate_textual_explanation(user_id, movie_row):
    """
    Generates human-readable explanation for a recommendation.
    """
    movie_genres = movie_row["genres"].split("|")

    # New user
    if user_id is None:
        return (
            "These movies are recommended because they are highly rated "
            "and popular among users."
        )

    # Existing user
    preferred_genres = get_user_genre_profile(user_id)

    matched_genres = set(preferred_genres).intersection(movie_genres)

    if matched_genres:
        return (
            f"Recommended because you often watch and rate "
            f"{', '.join(matched_genres)} movies."
        )

    return (
        "Recommended based on your overall viewing patterns "
        "and similar users' preferences."
    )

def precision_recall_at_k(user_id, k=10, test_ratio=0.2):
    user_items = ratings[ratings["userId"] == user_id]["movieId"].values

    if len(user_items) < 20:
        return None, None

    np.random.shuffle(user_items)

    split = int(len(user_items) * (1 - test_ratio))
    test_items = set(user_items[split:])

    # Allow watched movies during evaluation
    recommended = personalized_top_n(
        user_id, k, remove_watched=False
    )["movieId"].values

    recommended = set(recommended)

    tp = len(recommended & test_items)

    precision = tp / k
    recall = tp / len(test_items)

    return precision, recall


