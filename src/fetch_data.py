import yfinance as yf
import pandas as pd
import pathlib

def fetch_all_data(data_dir="data"):
    """
    Downloads full historical data for PLTR and ^IXIC using yfinance.
    Saves them to PLTR_current.csv and IXIC_current.csv in data_dir.
    """
    data_path = pathlib.Path(data_dir)
    data_path.mkdir(exist_ok=True, parents=True)

    print("Downloading latest data from Yahoo Finance...")

    # 1. Palantir
    print("Fetching PLTR...")
    pltr = yf.Ticker("PLTR")
    pltr_hist = pltr.history(period="max")
    # yfinance returns index as Datetime, reset to get 'Date' column
    pltr_hist = pltr_hist.reset_index()
    # Format Date to YYYY-MM-DD for consistency
    pltr_hist["Date"] = pltr_hist["Date"].dt.strftime("%Y-%m-%d")
    
    pltr_file = data_path / "PLTR_current.csv"
    pltr_hist.to_csv(pltr_file, index=False)
    print(f"Saved PLTR data to {pltr_file} ({len(pltr_hist)} rows)")

    # 2. NASDAQ (IVIC)
    print("Fetching ^IXIC (NASDAQ)...")
    ixic = yf.Ticker("^IXIC")
    ixic_hist = ixic.history(period="max")
    ixic_hist = ixic_hist.reset_index()
    ixic_hist["Date"] = ixic_hist["Date"].dt.strftime("%Y-%m-%d")

    ixic_file = data_path / "IXIC_current.csv"
    ixic_hist.to_csv(ixic_file, index=False)
    print(f"Saved NASDAQ data to {ixic_file} ({len(ixic_hist)} rows)")

if __name__ == "__main__":
    fetch_all_data()
