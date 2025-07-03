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

col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
with col1:
    user_input = st.number_input("Enter User ID", min_value=1, max_value=5000, step=1, value=st.session_state.get("user_input", 1), help="Type a user ID number from the dataset.")
with col2:
    confidence_threshold = st.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("confidence_threshold", 0.5),
        step=0.05,
        help="Only show similar users above this similarity score."
    )
with col3:
    language_codes = books['language_code'].dropna().unique().tolist()
    selected_languages = st.multiselect("Filter by Language Code", language_codes, help="Filter recommendations based on book language. This only affects the Recommended Books section.")

st.markdown("---")

# Refresh and Randomize buttons in one row
button_col2, button_col1 = st.columns([0.6, 0.6])
with button_col1:
    import time
    if "last_click" not in st.session_state:
        st.session_state["last_click"] = 0
    cooldown_seconds = 2
    now = time.time()
    can_click = now - st.session_state["last_click"] > cooldown_seconds
    refresh_disabled = not can_click or (user_input == st.session_state.get("previous_user") and confidence_threshold == st.session_state.get("previous_confidence"))
    if refresh_disabled:
        st.warning("⚠️ Change User ID or Similarity Score to enable refreshing recommendations.", icon="⚠️")
    refresh_clicked = st.button("Refresh Recommendations", help="Click once to update based on selected filters. Make sure to change User ID or Similarity Score first.", disabled=refresh_disabled)
    if refresh_clicked:
        st.session_state["last_click"] = now
with button_col2:
    import random
    if st.button("Randomize Inputs", key="random_inputs_button"):
        st.session_state["user_input"] = random.randint(1, 5000)
        st.session_state["confidence_threshold"] = round(random.uniform(0.0, 1.0), 2)
        st.rerun()
        st.session_state["user_input"] = random.randint(1, 5000)
        st.session_state["confidence_threshold"] = round(random.uniform(0.0, 1.0), 2)
        st.rerun()

if refresh_clicked and (user_input != st.session_state.get("previous_user") or confidence_threshold != st.session_state.get("previous_confidence")):
    st.session_state["previous_user"] = user_input
    st.session_state["previous_confidence"] = confidence_threshold
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
        st.markdown("These are the five most similar users to the selected user (if there are similar users), based on collaborative filtering and cosine similarity of their book rating patterns.")
        for i, uid in enumerate(top_users):
            st.markdown(f"<span style='color:#bbb;font-size:16px'>👤 <strong>User {uid}</strong> — Similarity Rank #{i+1}</span>", unsafe_allow_html=True)

        user_ratings = pivot_table.loc[user_input]
        unrated_books = user_ratings[user_ratings == 0].index
        avg_ratings = pivot_table.loc[top_users, unrated_books].mean()

        if selected_languages:
            book_meta = books[['book_id', 'language_code']]
            filtered_books = book_meta[book_meta['language_code'].isin(selected_languages)]['book_id']
            avg_ratings = avg_ratings[avg_ratings.index.isin(filtered_books)]

        top_recs = avg_ratings.sort_values(ascending=False).head(5)
        st.subheader("Recommended Books")
        st.markdown("These recommendations are predicted based on books rated highly by users with similar preferences, filtered optionally by language.")

        # Enrich top recommended books with metadata
        rec_books = books[books['book_id'].isin(top_recs.index)][['book_id', 'title', 'authors', 'original_publication_year', 'average_rating']]
        rec_books = rec_books.set_index('book_id').loc[top_recs.index]  # Keep top rec order
        rec_books["Estimated Rating"] = top_recs.values
        st.dataframe(rec_books)

        user_history = ratings[ratings['user_id'] == user_input].sort_values('book_id')
        if len(user_history) >= 5:
            st.subheader("Rating Trends")
            st.markdown("A comparison of the selected user’s past ratings with the predictions made by the trained machine learning model.")
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

            st.subheader("Model Comparison")
            st.markdown("""
            This section compares forecasting performance across two ML models: **Random Forest** and **Linear Regression**. This shows how different algorithms affect predictions.
            """)
            from sklearn.metrics import mean_squared_error
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_preds = lr_model.predict(X)
            rf_rmse = np.sqrt(mean_squared_error(y, pred))
            lr_rmse = np.sqrt(mean_squared_error(y, lr_preds))
            st.write(pd.DataFrame({"Model": ["Random Forest", "Linear Regression"], "RMSE": [rf_rmse, lr_rmse]}))

            st.subheader("Top Predicted Books")
            st.markdown("""
            This shows the books with the highest predicted ratings for the user, inferred from their past preferences. This is a proxy for feature influence when only one input feature (book ID) is used.
            """)
            importance_df = pd.DataFrame({
                "Book ID": X['book_id'],
                "Predicted Rating": pred
            }).copy()

            importance_df["Title"] = books.set_index("book_id").loc[importance_df["Book ID"]]["title"].values
            importance_df = importance_df.sort_values("Predicted Rating", ascending=False).drop_duplicates("Book ID").head(5)
            st.dataframe(importance_df[["Book ID", "Title", "Predicted Rating"]])

            st.subheader("User Clustering")
            st.markdown("""
            This demonstrates unsupervised learning by assigning the selected user to a cluster of similar users, based on SVD embeddings.
            Below is a t-SNE projection showing clusters in 2D space.
            """)
            from sklearn.manifold import TSNE
            import seaborn as sns
            import matplotlib.pyplot as plt

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_result = tsne.fit_transform(embeddings)
            cluster_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
            cluster_df['cluster'] = cluster_labels
            cluster_df['selected'] = [i == user_index for i in range(len(cluster_df))]

            fig_tsne, ax_tsne = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=cluster_df, x='x', y='y', hue='cluster', palette='tab10', style='selected', s=100, ax=ax_tsne)
            plt.title("t-SNE Visualization of User Clusters", fontsize=12)
            st.pyplot(fig_tsne)

            st.write(f"Selected user belongs to **Cluster {user_cluster}**")

            st.subheader("Recommendation Diversity")
            st.markdown("""
            A simple analysis of author diversity and rating variance among the recommended books.
            """)
            diversity_score = rec_books['authors'].nunique()
            rating_std = rec_books['average_rating'].std()
            st.write(f"- Authors represented: {diversity_score}")
            st.write(f"- Rating standard deviation: {rating_std:.2f}")

            st.subheader("Downloadable Report")
            st.markdown("""
            You can export your recommendations and insights into a downloadable CSV file.
            """)
            csv_data = rec_books.to_csv().encode('utf-8')
            st.download_button("📥 Download Recommendations CSV", data=csv_data, file_name="recommendations.csv", mime="text/csv")
