import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.utils import (
    load_ticker, compute_RSI, compute_MACD, compute_bollinger_width,
    compute_ROC, compute_ATR, compute_stochastic_k, make_sequences
)
from src.model import PalantirLSTM

# Constants
LOOKBACK = 60
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 50

def train(data_dir, output_dir):
    data_path = pathlib.Path(data_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. Load Data
    pltr_path = data_path / "PLTR_2025-12-04.csv"
    ixic_path = data_path / "IXIC_2025-12-04.csv"

    if not pltr_path.exists() or not ixic_path.exists():
        print(f"Error: Data files not found in {data_dir}")
        print(f"Expected: {pltr_path} and {ixic_path}")
        print("Please place the CSV files in the data directory.")
        return

    print("Loading data...")
    pltr_df, _ = load_ticker(pltr_path)
    nasdaq_df, _ = load_ticker(ixic_path)

    print("Processing features...")
    pltr_df["Date"] = pd.to_datetime(pltr_df["Date"])
    nasdaq_df["Date"] = pd.to_datetime(nasdaq_df["Date"])
    pltr_df = pltr_df.sort_values("Date").reset_index(drop=True)
    nasdaq_df = nasdaq_df.sort_values("Date").reset_index(drop=True)

    # Merge
    merged = pltr_df.merge(
        nasdaq_df[["Date", "Close", "Volume"]].rename(
            columns={"Close": "NAS_Close", "Volume": "NAS_Volume"}
        ),
        on="Date",
        how="inner"
    )

    # Calculate Technical Indicators
    merged["MA_5"]  = merged["Close"].rolling(window=5).mean()
    merged["MA_10"] = merged["Close"].rolling(window=10).mean()
    merged["MA_20"] = merged["Close"].rolling(window=20).mean()


    merged["RSI_14"] = compute_RSI(merged["Close"])
    merged["MACD"], merged["MACD_signal"], merged["MACD_hist"] = compute_MACD(merged["Close"])
    merged["BB_width"] = compute_bollinger_width(merged["Close"])
    merged["ROC_10"]   = compute_ROC(merged["Close"])
    merged["ATR_14"]   = compute_ATR(merged)
    merged["Stoch_K"]  = compute_stochastic_k(merged)

    # Add market context (NASDAQ correlations)
    merged["NAS_ret_1"] = merged["NAS_Close"].pct_change(1)
    merged["NAS_ret_5"] = merged["NAS_Close"].pct_change(5)

    merged = merged.dropna().reset_index(drop=True)

    # Feature columns
    feature_cols = [
        "Close", "Volume", "MA_5", "MA_10", "MA_20",
        "RSI_14", "MACD", "MACD_signal", "MACD_hist",
        "BB_width", "ROC_10", "ATR_14", "Stoch_K",
        "NAS_Close", "NAS_Volume", "NAS_ret_1", "NAS_ret_5"
    ]

    features = merged[feature_cols].values
    close_vals = merged["Close"].values.reshape(-1, 1)

    # Train/Test Split
    split_idx = len(merged) - 100
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    train_close = close_vals[:split_idx]
    test_close = close_vals[split_idx:]

    # Scales
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler = MinMaxScaler(feature_range=(0, 1))

    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)
    train_close_scaled = close_scaler.fit_transform(train_close)
    test_close_scaled = close_scaler.transform(test_close)

    # Save scalers
    joblib.dump(feature_scaler, output_path / "feature_scaler.pkl")
    joblib.dump(close_scaler, output_path / "close_scaler.pkl")
    print(f"Scalers saved to {output_path}")

    # Build Sequences
    X_train, y_train_sc, y_train_updown = make_sequences(train_features_scaled, train_close_scaled, train_close, LOOKBACK)
    X_test, y_test_sc, y_test_updown = make_sequences(test_features_scaled, test_close_scaled, test_close, LOOKBACK)

    # Convert to Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr_price = torch.tensor(y_train_sc, dtype=torch.float32)
    ytr_updown = torch.tensor(y_train_updown, dtype=torch.float32)

    train_ds = TensorDataset(Xtr, ytr_price, ytr_updown)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = PalantirLSTM(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.2
    ).to(device)

    criterion_price = nn.MSELoss()
    criterion_updown = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for Xb, yb_price, yb_up in train_loader:
            Xb = Xb.to(device)
            yb_price = yb_price.to(device)
            yb_up = yb_up.to(device)

            optimizer.zero_grad()
            pred_price, pred_up = model(Xb)

            loss_price = criterion_price(pred_price, yb_price)
            loss_up = criterion_updown(pred_up, yb_up)
            
            loss = loss_price + loss_up
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Save Model
    model_path = output_path / "palantir_lstm.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Path to save checkpoints")
    args = parser.parse_args()

    train(args.data_dir, args.output_dir)
