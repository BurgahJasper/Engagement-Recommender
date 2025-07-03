# engagement_recommender/app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Engagement Recommender", layout="wide", initial_sidebar_state="expanded")
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

**Instructions:**
1. Use the controls to select a user, similarity threshold, and language.
2. Click **Refresh Recommendations** to generate suggestions and predictions.
3. Avoid rapidly clicking the button — computations take a few seconds.

**How it works:**
- **User ID**: Select a user to get personalized recommendations based on their historical ratings.
- **Minimum Similarity Score**: Adjust the threshold to control how similar other users must be to influence recommendations.
- **Language Filter**: Choose which book languages you're interested in to narrow down suggestions.

**Behind the scenes:**
- The app uses **Truncated SVD** to generate user embeddings and compute user similarity.
- Based on similar users' ratings, it suggests books you haven’t rated yet.
- It trains a **Random Forest model** to forecast your future book preferences.
""")

@st.cache_data
def load_data():
    ratings = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")
    books = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv")
    ratings = ratings[ratings['user_id'] <= 5000]  # Limit to 5000 users for performance
    return ratings, books

@st.cache_data
def generate_embeddings(pivot_table):
    # Perform SVD only once and cache it
    svd = TruncatedSVD(n_components=20, random_state=42)
    return svd.fit_transform(pivot_table)

# Sidebar description for users
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This interactive recommender system was built using the Goodbooks-10K dataset.

    **Purpose:**
    - Demonstrate collaborative filtering using SVD (latent embeddings)
    - Forecast user behavior using machine learning
    - Allow filter-based exploration of book recommendations

    **Technologies used:**
    - Streamlit for interactive UI
    - Scikit-learn for ML and recommendations
    - Pandas & NumPy for data handling
    - Random Forest for forecasting future ratings

    Developed by Jasper Maximo Garcia
    """)

ratings, books = load_data()

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    user_input = st.number_input("Enter User ID", min_value=1, max_value=5000, step=1, help="Type a user ID number from the dataset.")
with col2:
    confidence_threshold = st.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show similar users above this similarity score."
    )
with col3:
    language_codes = books['language_code'].dropna().unique().tolist()
    selected_languages = st.multiselect("Filter by Language Code", language_codes, help="Filter recommendations based on book language.")

st.markdown("---")

# Refresh button to apply filters only on demand
if st.button("Refresh Recommendations", help="Click once to update based on selected filters."):
    with st.spinner("Generating recommendations..."):

            pivot_table = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

    if user_input in pivot_table.index:
        embeddings = generate_embeddings(pivot_table)  # Cached embedding computation
        cosine_sim = cosine_similarity(embeddings)

        user_index = pivot_table.index.get_loc(user_input)
        sim_scores = list(enumerate(cosine_sim[user_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Avoid recomputing and crashing when similarity score excludes too many users
        top_users = [pivot_table.index[i[0]] for i in sim_scores[1:] if sim_scores[i[0]][1] >= confidence_threshold][:5]

        st.subheader("Top Similar Users")
        st.write(top_users)

        user_ratings = pivot_table.loc[user_input]
        unrated_books = user_ratings[user_ratings == 0].index
        avg_ratings = pivot_table.loc[top_users, unrated_books].mean()

        if selected_languages:
            book_meta = books[['book_id', 'language_code']]
            filtered_books = book_meta[book_meta['language_code'].isin(selected_languages)]['book_id']
            avg_ratings = avg_ratings[avg_ratings.index.isin(filtered_books)]

        top_recs = avg_ratings.sort_values(ascending=False).head(5)
        st.subheader("Recommended Books")

        # Enrich top recommended books with metadata
        rec_books = books[books['book_id'].isin(top_recs.index)][['book_id', 'title', 'authors', 'original_publication_year', 'average_rating']]
        rec_books = rec_books.set_index('book_id').loc[top_recs.index]  # Keep top rec order
        rec_books["Estimated Rating"] = top_recs.values
        st.dataframe(rec_books)

        user_history = ratings[ratings['user_id'] == user_input].sort_values('book_id')
        if len(user_history) >= 5:
            st.subheader("Rating Trends")
            X = user_history[['book_id']]
            y = user_history['rating']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            pred = model.predict(X)

            book_titles = books.set_index('book_id').loc[user_history['book_id']]['title']
            chart_data = pd.DataFrame({"Actual Ratings": y.values, "Predicted Ratings": pred}, index=book_titles)
            st.line_chart(chart_data)

            st.subheader("Forecasted Ratings")

            # Show explanation of what forecasting means
            st.markdown("""
            These ratings are predictions of how this user might rate unseen books in the future,
            based on their past behavior and using a trained Random Forest model.

            Higher values (closer to 5) mean the model believes the user is likely to enjoy similar books.
            """)

            future_books = pd.DataFrame({'book_id': range(user_history['book_id'].max() + 1, user_history['book_id'].max() + 6)})
            future_preds = model.predict(future_books)
            future_titles = books.set_index('book_id').reindex(future_books['book_id'])['title'].fillna('Unknown Title')

            fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
            pd.Series(future_preds, index=future_titles, name="Forecasted Ratings").plot(kind='bar', ax=ax2, color='skyblue')
            ax2.set_facecolor('#0e1117')
            ax2.tick_params(colors='white')
            ax2.yaxis.label.set_color('white')
            ax2.xaxis.label.set_color('white')
            ax2.set_xlabel("")
            ax2.set_ylabel("Predicted Rating")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            ax2.set_title("Predicted Ratings for Unseen Books", color='white')
            st.pyplot(fig2)
