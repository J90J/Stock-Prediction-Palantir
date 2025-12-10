import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.model import PalantirLSTM
from src.utils import (
    load_ticker, compute_RSI, compute_MACD, compute_bollinger_width,
    compute_ROC, compute_ATR, compute_stochastic_k
)

# Constants (Must match training)
LOOKBACK = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURE_COLS = [
    "Close", "Volume", "MA_5", "MA_10", "MA_20",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "ROC_10", "ATR_14", "Stoch_K",
    "NAS_Close", "NAS_Volume", "NAS_ret_1", "NAS_ret_5"
]

def run_demo(data_dir, checkpoints_dir, results_dir):
    data_path = pathlib.Path(data_dir)
    checkpoints_path = pathlib.Path(checkpoints_dir)
    results_path = pathlib.Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Check for resources
    model_path = checkpoints_path / "palantir_lstm.pth"
    feature_scaler_path = checkpoints_path / "feature_scaler.pkl"
    close_scaler_path = checkpoints_path / "close_scaler.pkl"
    pltr_path = data_path / "PLTR_2025-12-04.csv"
    ixic_path = data_path / "IXIC_2025-12-04.csv"

    missing = []
    if not model_path.exists(): missing.append(str(model_path))
    if not feature_scaler_path.exists(): missing.append(str(feature_scaler_path))
    if not pltr_path.exists(): missing.append(str(pltr_path))

    if missing:
        print("Error: Missing required files to run demo.")
        print("Missing:", missing)
        print("Please train the model first by moving your data to 'data/' and running 'python src/main.py'")
        return

    # Update local data cache
    print("Auto-updating data...")
    from src.fetch_data import fetch_all_data
    from src.sentiment import get_current_sentiment
    
    # Optional: You can comment this out if you suspect rate limits or want to use cached data
    fetch_all_data(str(data_path))

    # Point to the NEW auto-downloaded files
    pltr_path = data_path / "PLTR_current.csv"
    ixic_path = data_path / "IXIC_current.csv"
    
    if not pltr_path.exists():
        print("Error: Data download failed.")
        return

    print("Loading resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Scalers
    feature_scaler = joblib.load(feature_scaler_path)
    
    # Load Data
    print("Processing latest data...")
    pltr_df, _ = load_ticker(pltr_path)
    nasdaq_df, _ = load_ticker(ixic_path)

    # Feature Engineer
    pltr_df["Date"] = pd.to_datetime(pltr_df["Date"])
    nasdaq_df["Date"] = pd.to_datetime(nasdaq_df["Date"])
    pltr_df = pltr_df.sort_values("Date").reset_index(drop=True)
    nasdaq_df = nasdaq_df.sort_values("Date").reset_index(drop=True)

    merged = pltr_df.merge(
        nasdaq_df[["Date", "Close", "Volume"]].rename(columns={"Close": "NAS_Close", "Volume": "NAS_Volume"}),
        on="Date", how="inner"
    )

    merged["MA_5"]  = merged["Close"].rolling(window=5).mean()
    merged["MA_10"] = merged["Close"].rolling(window=10).mean()
    merged["MA_20"] = merged["Close"].rolling(window=20).mean()
    merged["RSI_14"] = compute_RSI(merged["Close"])
    merged["MACD"], merged["MACD_signal"], merged["MACD_hist"] = compute_MACD(merged["Close"])
    merged["BB_width"] = compute_bollinger_width(merged["Close"])
    merged["ROC_10"]   = compute_ROC(merged["Close"])
    merged["ATR_14"]   = compute_ATR(merged)
    merged["Stoch_K"]  = compute_stochastic_k(merged)
    merged["NAS_ret_1"] = merged["NAS_Close"].pct_change(1)
    merged["NAS_ret_5"] = merged["NAS_Close"].pct_change(5)
    merged = merged.dropna().reset_index(drop=True)

    # Take last LOOKBACK days
    if len(merged) < LOOKBACK:
        print(f"Not enough data. Need {LOOKBACK} rows, have {len(merged)}")
        return

    last_segment = merged.iloc[-LOOKBACK:]
    last_close = last_segment["Close"].iloc[-1]
    last_date = last_segment["Date"].iloc[-1]

    # Prepare input
    features = last_segment[FEATURE_COLS].values
    features_scaled = feature_scaler.transform(features)
    
    X_input = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Load Model
    model = PalantirLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Running prediction for date after {last_date.date()}...")
    with torch.no_grad():
        pred_ret, pred_up = model(X_input)

    pred_ret_val = float(pred_ret.cpu().numpy()[0, 0])
    p_up = float(pred_up.cpu().numpy()[0, 0])
    model_verdict = "UP" if p_up >= 0.5 else "DOWN"
    next_close_pred = last_close * (1.0 + pred_ret_val)

    
    
    # Get News Sentiment
    sentiment = get_current_sentiment("PLTR")
    news_score = sentiment['score']
    news_verdict = sentiment['verdict']

    # Combine Model Prob (0-1) and Sentiment Score (-1 to 1)
    # Simple Heuristic: If they agree, boost confidence.
    
    final_recommendation = "HOLD"
    
    if model_verdict == "UP" and news_verdict == "POSITIVE":
        final_recommendation = "STRONG BUY 🟢"
    elif model_verdict == "DOWN" and news_verdict == "NEGATIVE":
        final_recommendation = "STRONG SELL 🔴"
    elif model_verdict == "UP":
        final_recommendation = "BUY (Model Bullish, News Neutral/Mixed) 🔵"
    elif model_verdict == "DOWN":
        final_recommendation = "SELL (Model Bearish, News Neutral/Mixed) 🟠"

    # Print Report
    print("\n" + "="*50)
    print(f"     PALANTIR (PLTR) HYBRID REPORT     ")
    print("="*50)
    print(f"Last Close Price:   ${last_close:.2f}")
    print(f"Predicted Return:   {pred_ret_val*100:.2f}%")
    print(f"Predicted Price:    ${next_close_pred:.2f}")
    print("-" * 30)
    print(f"[MODEL] Technicals: {model_verdict} (Confidence: {p_up*100:.1f}%)")
    print(f"[NEWS]  Sentiment:  {news_verdict} (Score: {news_score:.2f})")
    print("  Top Headlines:")
    for h in sentiment['headlines']:
        print(f"  - {h}")
    print("-" * 30)
    print(f"FINAL VERDICT:      {final_recommendation}")
    print("="*50 + "\n")

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(last_segment["Date"], last_segment["Close"], label="History (Last 60 Days)", color='blue')
    plt.scatter(last_date + pd.Timedelta(days=1), next_close_pred, color="red", label="Prediction", marker="x", s=150, zorder=5)
    
    title_text = f"Pred: ${next_close_pred:.2f} | Model: {model_verdict} | News: {news_verdict}"
    plt.title(f"Palantir Hybrid Forecast\n{title_text}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = results_path / "prediction_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_demo(args.data_dir, args.checkpoints_dir, args.results_dir)
