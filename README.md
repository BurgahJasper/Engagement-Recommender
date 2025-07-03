# ML Powered Engagement Recommender

This Streamlit app uses filtering and machine learning to recommend books and forecast user preferences from a database. It is built using the [Goodbooks-10K dataset](https://github.com/zygmuntz/goodbooks-10k) and includes advanced recommendation and forecasting techniques.

## Features

- Personalized Recommendations
Input a User ID (from 0-5000) to receive tailored recommendations based on historical rating behavior.

- Collaborative Filtering with SVD + Cosine Similarity
Uses TruncatedSVD (latent embeddings) to represent users in compressed vector spaces and find the most similar users via cosine similarity.

- Minimum Similarity Score Control
Adjust a threshold slider to control how similar users must be to influence recommendations.

- Language Filtering
Filter recommendations based on the book’s language code (applies only to the "Recommended Books" section).

- Randomize Inputs
Instantly try new inputs with one click using the "Randomize Inputs" button.

- Cooldown and Input Validation
Prevents spamming by disabling the refresh button until a new user ID or similarity threshold is selected. A tooltip and warning message guide proper usage.

- Recommended Books Section
Shows books highly rated by similar users that the selected user hasn't yet rated.

- Rating Trends Visualization
Plots a comparison between the user's actual past ratings and predicted values using a trained Random Forest model.

- Forecasted Ratings
Uses Random Forest to predict ratings for unseen books. Visualized as a horizontal bar chart with book titles.

- Model Comparison
Evaluates forecasting accuracy by comparing RMSE between:

Random Forest (flexible and accurate)

Linear Regression (simple baseline)
RMSE (Root Mean Squared Error) is reported to help assess model performance.

- Top Predicted Books
Displays the five books with the highest predicted ratings for the user, based on their past behavior. This differs from forecasted ratings by focusing on books already seen.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **ML & Data:** `pandas`, `numpy`, `scikit-learn`, `TruncatedSVD`, `RandomForestRegressor`
- **Visualization:** Streamlit's built-in charts, `matplotlib`
- **Deployment:** [Streamlit Cloud](https://streamlit.io/cloud)

## Installation

```bash
git clone https://github.com/your-username/engagement-recommender.git
cd engagement-recommender
pip install -r requirements.txt
streamlit run app.py
