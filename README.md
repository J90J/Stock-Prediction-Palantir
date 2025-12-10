# Palantir Stock Prediction with LSTM

## Introduction
For my final project, I explored whether a "Hybrid" approach—combining traditional Deep Learning with modern Sentiment Analysis—could better predict Palantir's (PLTR) stock price. Palantir is often driven by retail interactions and news cycles as much as technical fundamentals, so verifying standard technicals against news sentiment felt like a necessary step for accurate prediction.

My system uses an LSTM neural network to analyze 60 days of price/volume history to determine the technical trend, and then cross-references that with real-time news sentiment (analyzed via VADER) to generate a final trading recommendation.

## Challenges & Design Decisions
One of the biggest hurdles was data availability. While basic price data is free, obtaining historical news data for training is prohibitively expensive.
*   **Hybrid Inference**: Since I couldn't train the model on years of news history, I opted for a hybrid inference approach. The model predicts the *Technical* direction, and the Sentiment Engine acts as a "filter" or "confirming signal" based on *today's* news.
*   **Sentiment Granularity**: I chose VADER over transformer-based models (like BERT) because financial headlines are often short and direct, where VADER's rule-based approach performs efficiently without needing a heavy GPU for just text processing.

## Directory Structure
```
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── src/
│   ├── main.py         # Training script
│   ├── model.py        # LSTM model definition
│   ├── utils.py        # Helper functions (data loading, indicators)
├── data/               # Place your CSV data files here
├── checkpoints/        # Saved models and scalers
├── demo/
│   ├── demo.py         # Demo script for inference
└── results/            # Generated plots and predictions
```

## New Key Features
*   **Automated Data Fetching**: No need to manually download CSVs. The system grabs the latest data from `yfinance`.
*   **News Sentiment Engine**: Analyzes real-time news headlines to determine market sentiment (Positive/Negative).
*   **Hybrid "Smart Advisor"**: Combines the LSTM model's technical prediction with news sentiment for a final "Strong Buy/Sell" recommendation.

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Demo**:
    The demo script handles everything (data downloading + prediction + sentiment analysis).
    ```bash
    python demo/demo.py
    ```
    This will:
    *   **Download** the latest Palantir and NASDAQ data automatically.
    *   **Run Inference** using the trained LSTM model.
    *   **Analyze News** headlines for sentiment.
    *   **Generate Report** with a final verdict and plot.

## Training (Optional)
If you want to re-train the model yourself:
1.  Download fresh data:
    ```bash
    python src/fetch_data.py
    ```
2.  Run the training script:
    ```bash
    python src/main.py
    ```

*   **Hyperparameters**:
    *   Lookback Window: 60 days
    *   Hidden Size: 64
    *   Layers: 2
    *   Dropout: 0.2
    *   Learning Rate: 0.001
    *   Epochs: 50
    *   Batch Size: 32
*   **Random Seed**: While PyTorch seeds are not explicitly fixed in this version, the training process is stable. For strict determinism, set torch/numpy seeds at the start of `src/main.py`.

## Model Information
*   **Architecture**: Multi-layer LSTM with two heads:
    1.  **Regression Head**: Predicts the continuous return.
    2.  **Classification Head**: Predicts the probability of the price moving UP.
*   **Features**: OHLCV data + Technical Indicators (RSI, MACD, BB, ROC, ATR, Stochastic Oscillator) + NASDAQ Index correlations.

## Acknowledgments
*   Data sourced from Yahoo Finance.
*   Built with PyTorch.
