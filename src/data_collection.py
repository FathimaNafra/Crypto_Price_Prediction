import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import COINS, START_DATE
import matplotlib.pyplot as plt
import time

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

# CoinGecko API mapping for coins
COINGECKO_IDS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana",
    "XRP-USD": "ripple"
}

def fetch_data_coingecko():
    """Fetch historical price data from CoinGecko API"""
    for coin in COINS:
        coin_id = COINGECKO_IDS.get(coin)
        if not coin_id:
            print(f"Skipping {coin}: CoinGecko ID not found")
            continue
        
        print(f"Downloading {coin} from CoinGecko...")
        
        # CoinGecko API endpoint for historical data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "max",
            "interval": "daily"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse prices (returns [timestamp, price])
            prices = data["prices"]
            
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
            df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df = df.drop("Timestamp", axis=1)
            df.set_index("Date", inplace=True)
            
            # Filter by start date
            start = pd.to_datetime(START_DATE)
            df = df[df.index >= start]
            
            # Save to CSV
            df.to_csv(RAW_PATH / f"{coin}.csv", index_label="Date")
            print(f"✓ Saved {coin}: {len(df)} records")
            
            # Be nice to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ Error downloading {coin}: {e}")

def fetch_data():
    """Fetch data using yfinance (legacy method)"""
    for coin in COINS:
        print(f"Downloading {coin}...")
        df=yf.download(coin, start=START_DATE)
        df.to_csv(RAW_PATH / f"{coin}.csv", index_label="Date")


def plot_close(coin):
        df=yf.download(coin, start=START_DATE)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index,df["Close"])
        plt.title(f"{coin} Closing Prices")
        plt.show()

if __name__ == "__main__":
    # Use CoinGecko API (recommended - no API key needed)
    fetch_data_coingecko()
    
   
    # Plot the downloaded data
    plot_close("BTC-USD")
    plot_close("ETH-USD")
    plot_close("BNB-USD")
    plot_close("SOL-USD")
    plot_close("XRP-USD")