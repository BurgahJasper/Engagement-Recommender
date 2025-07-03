# ML Powered Engagement Recommender

This Streamlit app recommends books and forecasts user preferences using collaborative filtering, machine learning, and neural network models. It is built using the [Goodbooks-10K dataset](https://github.com/zygmuntz/goodbooks-10k) and enables interactive exploration of engagement trends and model predictions.

[Live Demo](https://jasper-engagement-recommender.streamlit.app/)

## Features

- Personalized Recommendations
Enter a User ID (1–5000) to get tailored book recommendations based on historical rating behavior.

- Collaborative Filtering with SVD + Cosine Similarity
Uses Truncated SVD to generate latent embeddings, allowing cosine similarity to match users with similar taste.

- Similarity Score Threshold
Customize a similarity threshold slider to filter how closely other users must match before influencing recommendations.

- Language Filtering
Filter book recommendations by language code (applies to the “Recommended Books” section only).

- Input Randomization
Click the Randomize Inputs button to instantly explore recommendations for a new user and similarity setting.

- Smart Refresh Controls
Prevents redundant refreshes by disabling the button until inputs change. Tooltips and warnings help guide interaction.

- Recommended Books
Displays the top-rated books by similar users that your selected user hasn’t read yet.

- Rating Trends
Plots actual user ratings vs. predictions from a trained PyTorch neural network.

- Forecasted Ratings
Predicts how your user might rate unseen books using a PyTorch model. Visualized in a horizontal bar chart.

- Model Comparison (Forecasting)
Compare RMSE across two forecasting models:

Random Forest (nonlinear, more expressive)

Linear Regression (simple baseline)

Displays performance as both markdown and a table.

- Embedding Comparison Chart
Toggle to compare predictions from:

Random Forest with SVD Embeddings

Random Forest with Raw Book IDs
Visually demonstrates the benefit of embeddings using a side-by-side line chart.

- Top Predicted Books
Shows the five books with the highest predicted ratings (based on the neural model). Reflects the strongest confidence in prior reading behavior.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **ML & Data:** `PyTorch`, `pandas`, `numpy`, `scikit-learn`, `TruncatedSVD`, `RandomForestRegressor`
- **Visualization:** `matplotlib`, Streamlit's charting
- **Deployment:** [Streamlit Cloud](https://streamlit.io/cloud)

## Installation

```bash
git clone https://github.com/your-username/engagement-recommender.git
cd engagement-recommender
pip install -r requirements.txt
streamlit run app.py
