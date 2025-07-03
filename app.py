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
- It trains a **Random Forest model** and a **PyTorch neural network** to forecast your future book preferences.
- **Top Predicted Books** shows predicted ratings for books already rated.
- **Forecasted Ratings** predicts ratings for new/unseen books.
- **Training Loss Curve** visualizes learning over time for the PyTorch model.
""")

compare_embedding = st.toggle("Compare With vs. Without Embedding", help="Shows how predictions differ with vs. without SVD embedding layers.")

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
            num_books = books['book_id'].max() + 1  # Total number of books for the embedding layer
            import torch
            import torch.nn as nn
            import torch.optim as optim

            class BookEmbeddingRegressor(nn.Module):
                def __init__(self, num_books, embed_dim=16, hidden_dim=32):
                    super().__init__()
                    self.embedding = nn.Embedding(num_books, embed_dim)
                    self.model = nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                    )

                def forward(self, x):
                    embedded = self.embedding(x)
                    return self.model(embedded).squeeze(-1)  # removes last dimension

            def train_model(X_train, y_train, num_books, epochs=300):
                X_tensor = torch.tensor(X_train.values, dtype=torch.long)
                y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

                model = BookEmbeddingRegressor(num_books=num_books, embed_dim=16, hidden_dim=32)
                criterion = nn.MSELoss()
                optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

                losses = []
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    losses.append(loss.item())

                return model, losses

            model, losses = train_model(X, y, num_books=num_books)
            X_tensor = torch.tensor(X.values, dtype=torch.long)
            pred = model(X_tensor).detach().numpy().flatten()

            book_titles = books.set_index('book_id').loc[user_history['book_id']]['title']
            chart_data = pd.DataFrame({"Actual Ratings": y.values, "Predicted Ratings": pred}, index=book_titles)
            st.line_chart(chart_data)

            st.subheader("Embedding Comparison Chart")
            st.markdown("This shows how predictions differ using a Random Forest model trained on SVD embeddings versus raw book IDs.")

            # Model using embeddings (SVD)
            # Generate embeddings from pivot table and map to books
            pivot_table = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
            svd = TruncatedSVD(n_components=20, random_state=42)
            book_embeddings = svd.fit_transform(pivot_table.T)

            book_id_to_index = {bid: idx for idx, bid in enumerate(pivot_table.columns)}
            book_indices = [book_id_to_index[bid] for bid in user_history['book_id'] if bid in book_id_to_index]
            X_embed = np.array([book_embeddings[idx] for idx in book_indices])

            rf_embed_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_embed_model.fit(X_embed, y)
            embed_preds = rf_embed_model.predict(X_embed)

            # Model without embeddings (raw book_id)
            X_raw = user_history['book_id'].values.reshape(-1, 1)
            rf_raw_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_raw_model.fit(X_raw, y)
            raw_preds = rf_raw_model.predict(X_raw)

            # Comparison chart
            compare_df = pd.DataFrame({
                "Actual Ratings": y.values,
                "With Embedding (SVD)": embed_preds,
                "Without Embedding (Raw Book ID)": raw_preds
            }, index=book_titles)

            st.line_chart(compare_df)

            st.subheader("Training Loss Curve")
            st.markdown("Visualizes how the PyTorch neural network improves its predictions over each training epoch.")
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0e1117')
            ax.plot(losses, color='lightblue')
            ax.set_title("PyTorch Neural Network Training Loss", color='white')
            ax.set_xlabel("Epoch", color='white')
            ax.set_ylabel("Loss", color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)

            st.subheader("Forecasted Ratings")
            st.markdown("""
            These ratings are predictions of how this user might rate **unseen books** in the future, based on their past behavior and a trained Random Forest model.

            This differs from the **Top Predicted Books** section, which shows high scores for books the user has already rated (or rated-like books).

            Forecasting shows expected interest in **brand new books**, while top predicted books reflect confidence in favorites.
            """)
            
            future_books = pd.DataFrame({'book_id': range(user_history['book_id'].max() + 1, user_history['book_id'].max() + 6)})
            future_tensor = torch.tensor(future_books['book_id'].values, dtype=torch.long)
            future_preds = model(future_tensor).detach().numpy().flatten()
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

            from sklearn.metrics import mean_squared_error
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_preds = lr_model.predict(X)
            rf_rmse = np.sqrt(mean_squared_error(y, pred))
            lr_rmse = np.sqrt(mean_squared_error(y, lr_preds))

            st.subheader("Model Comparison")
            st.markdown(f"""
            This section compares forecasting performance across two ML models: **Random Forest** and **Linear Regression**.

            The **Root Mean Squared Error (RMSE)** measures how closely each model's predictions align with the actual ratings:
            - A **lower RMSE** means the model makes more accurate predictions.
            - In this case:
              - Random Forest RMSE: **{rf_rmse:.4f}** — captures complex user behavior well
              - Linear Regression RMSE: **{lr_rmse:.4f}** — simpler, but less flexible

            These scores help decide which model is more suitable for forecasting ratings.
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
            st.dataframe(importance_df[["Book ID", "Title", "Predicted Rating"]].reset_index(drop=True))
