# 🎬 AI Movie Recommendation System with Explainable AI

An interactive Hybrid Movie Recommendation System built using collaborative filtering, content-based filtering, global popularity ranking, and Explainable AI (SHAP).
The system supports both new users and existing users, includes a feedback loop, and provides model evaluation metrics.

## 🚀 Features

### 🔹 Hybrid Recommendation Engine

Global popularity-based recommendations (IMDB weighted rating)

User-based collaborative filtering (cosine similarity)

Genre-based content filtering

Personalized recommendations for existing users

Cold-start handling for new users

### 🔹 Explainable AI (XAI)

SHAP-based feature explanations

Textual explanations for recommendation reasoning

Visual waterfall plots showing feature contribution

### 🔹 Feedback Mechanism

Users can like 👍 or dislike 👎 recommendations

Feedback dynamically influences future suggestions

### 🔹 Evaluation Metrics

Precision@K

Recall@K

Train-test split evaluation per user

Proper separation of recommendation and evaluation logic

### 🔹 Interactive UI

Built using Streamlit

Supports:

    New users

    Existing users

    Genre-specific recommendations

    Evaluation on demand

## 🧠 Model Overview
1️⃣ Popularity Model

Uses IMDB Weighted Rating:
``` bash
WR = (v / (v + m)) * R + (m / (v + m)) * C
```
Where:

R = Average rating

v = Number of votes

m = Minimum vote threshold

C = Global average rating

2️⃣ Collaborative Filtering

User–user similarity using cosine similarity

Top similar users contribute to recommendation score

3️⃣ Explainability

Linear Regression trained on rating features

SHAP waterfall plots show feature contribution

## 📂 Project Structure

``` code
.
├── app.py                # Streamlit UI
├── recommender.py        # Recommendation logic & evaluation
├── ml-1m/                # MovieLens 1M dataset
│   ├── movies.dat
│   ├── ratings.dat
├── README.md
```
## 📊 Dataset

This project uses the MovieLens 1M Dataset:

    1 million ratings

    6,000 users

    4,000 movies

Dataset link: https://grouplens.org/datasets/movielens/

Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```
### 2️⃣ Install dependencies
```python
pip install -r requirements.txt
```
### 3️⃣ Run the application
```python
streamlit run app.py
```
## 📈 Evaluation

-User-level train-test split

-Precision@K

-Recall@K

-No data leakage during evaluation

## 🖥️ How It Works
#### New User

    -Can select preferred genres

    -Receives top-rated movies

    -SHAP explains why the top movie is recommended

#### Existing User

    Receives:

          -Global top movies

          -Personalized recommendations

           -Genre-based recommendations

    -Can provide feedback

    -Can evaluate recommendation performance

    -SHAP visualizes feature impact

## Author

Ch. Mounika

B.Tech – Computer Science IIITKottayam

Project: AI-Based Movie Recommendation System with Explainable AI
