# engagement_recommender/app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Engagement Recommender", layout="wide")
st.markdown("""
<style>
    body {
        color: #f0f0f0;
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("ML Powered Engagement Recommender")
st.markdown("""
This app uses collaborative filtering and machine learning to recommend books and forecast ratings.
Customize your input to explore personalized suggestions and trends.
""")

@st.cache_data
def load_data():
    ratings = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")
    books = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv")
    return ratings, books

ratings, books = load_data()

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    user_input = st.number_input("Enter User ID", min_value=1, step=1, help="Type a user ID number from the dataset.")
with col2:
    confidence_threshold = st.slider("Minimum Similarity Score", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Only show similar users above this similarity score.")
with col3:
    genres = books['genres'].dropna().unique().tolist() if 'genres' in books.columns else []
    selected_genres = st.multiselect("Filter by Genre", genres, help="Select preferred genres to filter recommendations.")

st.markdown("---")

if user_input:
    pivot_table = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

    if user_input in pivot_table.index:
        svd = TruncatedSVD(n_components=20, random_state=42)
        embeddings = svd.fit_transform(pivot_table)
        cosine_sim = cosine_similarity(embeddings)

        user_index = pivot_table.index.get_loc(user_input)
        sim_scores = list(enumerate(cosine_sim[user_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_users = [pivot_table.index[i[0]] for i in sim_scores[1:6] if sim_scores[i[0]][1] >= confidence_threshold]

        st.subheader("Top Similar Users")
        st.write(top_users)

        user_ratings = pivot_table.loc[user_input]
        unrated_books = user_ratings[user_ratings == 0].index
        avg_ratings = pivot_table.loc[top_users, unrated_books].mean()

        if selected_genres and 'genres' in books.columns:
            book_meta = books[['book_id', 'genres']]
            filtered_books = book_meta[book_meta['genres'].isin(selected_genres)]['book_id']
            avg_ratings = avg_ratings[avg_ratings.index.isin(filtered_books)]

        top_recs = avg_ratings.sort_values(ascending=False).head(5)
        st.subheader("📚 Recommended Books")
        st.dataframe(top_recs.rename("Estimated Rating"))

        user_history = ratings[ratings['user_id'] == user_input].sort_values('book_id')
        if len(user_history) >= 5:
            st.subheader("Rating Trends")
            X = user_history[['book_id']]
            y = user_history['rating']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            pred = model.predict(X)

            chart_data = pd.DataFrame({"Actual Ratings": y.values, "Predicted Ratings": pred}, index=user_history['book_id'])
            st.line_chart(chart_data)

            st.subheader("Forecasted Ratings")
            future_books = pd.DataFrame({'book_id': range(user_history['book_id'].max() + 1, user_history['book_id'].max() + 6)})
            future_preds = model.predict(future_books)
            st.bar_chart(pd.Series(future_preds, index=future_books['book_id'], name="Forecasted Ratings"))
    else:
        st.warning("User ID not found in dataset.")
