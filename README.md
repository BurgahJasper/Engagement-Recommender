# ML Powered Engagement Recommender

This Streamlit app uses collaborative filtering and machine learning to recommend books and forecast user preferences. It is built using the [Goodbooks-10K dataset](https://github.com/zygmuntz/goodbooks-10k) and includes advanced recommendation and forecasting techniques.

## Features

- Input a user ID to personalize recommendations.
- Uses TruncatedSVD (latent embeddings) and cosine similarity to find top similar users.
- Recommends books based on ratings from similar users.
- Forecasts future ratings using Random Forest Regressor.
- Filters recommendations by genre (if available).
- Adjusts recommendation sensitivity with a similarity score slider.
- Dark mode styling with tooltips for usability.

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
