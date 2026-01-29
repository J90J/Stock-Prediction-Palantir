
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
import pathlib
import sys

# --- Safe Import Setup ---
try:
    # Add project root to path so we can import from src
    sys.path.append(str(pathlib.Path(__file__).parent))
except Exception as e:
    st.error(f"System Path Error: {e}")

# Constants
LOOKBACK = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURE_COLS = [
    "Close", "Volume", "MA_5", "MA_10", "MA_20",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "ROC_10", "ATR_14", "Stoch_K",
    "NAS_Close", "NAS_Volume", "NAS_ret_1", "NAS_ret_5"
]

# --- Sidebar Integration ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/13/Palantir_Technologies_logo.svg", width=200)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
st.sidebar.info("This is the **Stock Prediction** module.")
if st.sidebar.button("Go to Text Search (Mini Project 1)"):
    st.sidebar.markdown("[Click here to open Text Search](https://mini-project-1-c99pgkfdngm9bkgjm93elc.streamlit.app/)")

st.sidebar.markdown("---")

# --- Main App ---
st.title("Palantir (PLTR) Stock Predictor 🚀")
st.markdown("Hybrid prediction using **LSTM Neural Networks** and **News Sentiment Analysis**.")

# Setup Paths
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

DATA_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# 1. Fetch Data
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🔄 Update Market Data"):
        with st.spinner("Downloading latest data from Yahoo Finance..."):
            try:
                from src.fetch_data import fetch_all_data
                fetch_all_data(str(DATA_DIR))
                st.success("Data updated!")
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")

# 2. Load Resources
@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_path = CHECKPOINTS_DIR / "palantir_lstm.pth"
    feature_scaler_path = CHECKPOINTS_DIR / "feature_scaler.pkl"
    close_scaler_path = CHECKPOINTS_DIR / "close_scaler.pkl"
    
    if not model_path.exists():
        return None, None, None, "Model file not found. Please train first."
        
    try:
        from src.model import PalantirLSTM
    except ImportError as e:
        return None, None, None, f"Import Error (src.model): {e}"

    # Load
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        close_scaler = joblib.load(close_scaler_path)
        
        model = PalantirLSTM(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        return None, None, None, f"Error loading model/scalers: {e}"
    
    return model, feature_scaler, device, None

model, feature_scaler, device, error = load_resources()

if error:
    st.error(error)
    st.stop()

# 3. Process & Predict
pltr_path = DATA_DIR / "PLTR_current.csv"
ixic_path = DATA_DIR / "IXIC_current.csv"

if not pltr_path.exists():
    st.warning("Data not found. Please click 'Update Market Data'.")
    st.stop()

# Load Data
try:
    from src.utils import (
        load_ticker, compute_RSI, compute_MACD, compute_bollinger_width,
        compute_ROC, compute_ATR, compute_stochastic_k
    )
    pltr_df, _ = load_ticker(pltr_path)
    nasdaq_df, _ = load_ticker(ixic_path)
except Exception as e:
    st.error(f"Error loading utils or data: {e}")
    st.stop()

# Preprocessing (Same as demo.py)
try:
    pltr_df["Date"] = pd.to_datetime(pltr_df["Date"])
    nasdaq_df["Date"] = pd.to_datetime(nasdaq_df["Date"])
    pltr_df = pltr_df.sort_values("Date").reset_index(drop=True)
    nasdaq_df = nasdaq_df.sort_values("Date").reset_index(drop=True)

    merged = pltr_df.merge(
        nasdaq_df[["Date", "Close", "Volume"]].rename(columns={"Close": "NAS_Close", "Volume": "NAS_Volume"}),
        on="Date", how="inner"
    )

    # Indicators
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

    # Prepare Input
    last_segment = merged.iloc[-LOOKBACK:]
    last_close = last_segment["Close"].iloc[-1]
    last_date = last_segment["Date"].iloc[-1]

    features = last_segment[FEATURE_COLS].values
    features_scaled = feature_scaler.transform(features)
    X_input = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_ret, pred_up = model(X_input)

    pred_ret_val = float(pred_ret.cpu().numpy()[0, 0])
    p_up = float(pred_up.cpu().numpy()[0, 0])
    model_verdict = "UP" if p_up >= 0.5 else "DOWN"
    next_close_pred = last_close * (1.0 + pred_ret_val)
except Exception as e:
    st.error(f"Prediction Logic Error: {e}")
    st.stop()

# Sentiment
st.subheader("📰 News Sentiment Analysis")
with st.spinner("Analyzing top headlines..."):
    try:
        from src.sentiment import get_current_sentiment
        sentiment = get_current_sentiment("PLTR")
        news_score = sentiment['score']
        news_verdict = sentiment['verdict']
    except Exception as e:
        st.error(f"Sentiment Analysis Failed: {e}")
        news_score = 0
        news_verdict = "ERROR"
        sentiment = {"headlines": []}

# Final Logic
final_recommendation = "HOLD"
if model_verdict == "UP" and news_verdict == "POSITIVE":
    final_recommendation = "STRONG BUY 🟢"
elif model_verdict == "DOWN" and news_verdict == "NEGATIVE":
    final_recommendation = "STRONG SELL 🔴"
elif model_verdict == "UP":
    final_recommendation = "BUY (Model Bullish) 🔵"
elif model_verdict == "DOWN":
    final_recommendation = "SELL (Model Bearish) 🟠"

# --- Dashboard ---
st.markdown("### 📊 Forecast Results")
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${last_close:.2f}")
col2.metric("Predicted Price", f"${next_close_pred:.2f}", f"{pred_ret_val*100:.2f}%")
col3.metric("Final Verdict", final_recommendation)

st.progress(p_up, text=f"Model Confidence (Bullish): {p_up*100:.1f}%")

# Plot
st.markdown("### 📈 Price Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(last_segment["Date"], last_segment["Close"], label="History (60 Days)", color='blue')
ax.scatter(last_date + pd.Timedelta(days=1), next_close_pred, color="red", label="Prediction", marker="x", s=150, zorder=5)
ax.set_title(f"Prediction for {last_date.date() + pd.Timedelta(days=1)}")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# News Display
st.markdown("### 🗞️ Top Headlines")
st.write(f"**Sentiment**: {news_verdict} (Score: {news_score:.2f})")
for h in sentiment['headlines']:
    st.markdown(f"- {h}")
