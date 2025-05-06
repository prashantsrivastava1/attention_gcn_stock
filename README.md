# Attention-Based BiLSTM Model for Stock Price Prediction and Influential Factor Identification

This repository contains the full implementation of a master's thesis project focused on developing and evaluating an attention-based BiLSTM model to identify influential factors driving stock prices, using Vedanta Limited (VEDL) as the case study.

## 📘 Project Overview

The core goal of this research is not just to predict stock prices but to interpret which input variables (like sentiment, GDP, inflation, and sector indices) influence price movements. Traditional forecasting models struggle with interpretability; this work addresses that gap using attention mechanisms on top of BiLSTM.

## 🧠 Model Architecture

The model is built with:

- **BiLSTM**: Captures temporal dependencies from historical stock data.
- **Attention Layer**: Highlights influential time steps and features.
- **Final Dense Layer**: Predicts stock closing prices.

GCN was initially proposed but removed due to poor performance during experimentation.

## 📁 Repository Structure

attention_gcn_stock/
├── data/ # Raw and preprocessed CSVs (merged data, sentiment, etc.)
├── models/
│ └── attention_lstm.py # Final BiLSTM + Attention model
├── train.py # Training script
├── evaluate.py # Evaluation script (MSE, MAE, R², and visualizations)
├── model_comparison.py # Script to compare multiple models (Linear, RF, XGB, LSTM, etc.)
├── generate_predictions.py # Save y_test and y_pred for final plots
├── figures/ # Saved PNG figures for thesis report
└── README.md # You’re reading it!

## 📊 Data Sources

- **VEDL historical stock data**: Yahoo Finance
- **Nifty Metal Index**: NSE India
- **GDP & Inflation**: Indian Government datasets
- **Sentiment Scores**: Extracted using NLP from news headlines via Google RSS Feeds

## 🔧 Requirements

```bash
pip install -r requirements.txt

🏁 How to Run
Train the Model:

bash
Copy
Edit
python train.py
Evaluate Performance:

bash
Copy
Edit
python evaluate.py
Compare with Other Models:

bash
Copy
Edit
python model_comparison.py
Generate Predictions (for visualization):

bash
Copy
Edit
python generate_predictions.py
📈 Key Metrics
Model	MSE	MAE	R²
Linear Regression	0.0047	0.0456	0.9952
XGBoost	0.0808	0.1995	0.9174
LSTM + Attention	0.5069	0.6563	0.4818

⚠️ Note: The attention-based model underperformed on pure regression but enabled feature attribution and interpretability — which was the core objective.

📌 Thesis Contribution
Novelty: Applies BiLSTM + attention to identify influential stock predictors, not just price.

Insight: Attention weights expose market-sentiment, economic indicators, and sectoral dependencies.

Application: Tools like heatmaps, attention graphs, and attribution plots are integrated.

🧾 License
This project is part of a Master of Science thesis submission and is intended for academic, non-commercial use.

Developed by Prashant Srivastava | MS (AI & ML), LJMU

yaml
Copy
Edit
```
