import pandas as pd
from pathlib import Path
from config import COINS
import os

# Make paths absolute based on script location
BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data/raw"
PROCESSED = BASE_DIR / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def preprocess(df):
    df["Daily_Return"] = df["Close"].pct_change()          
    df["MA_7"] = df["Close"].rolling(7).mean()            
    df["MA_30"] = df["Close"].rolling(30).mean()          
    df["Volatility"] = df["Daily_Return"].rolling(7).std()
    return df.dropna()                                    


def run():
    for coin in COINS:
        print(f"Processing {coin}...")
        file_path = RAW / f"{coin}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist. Skipping {coin}.")
            continue
        try:
            df = pd.read_csv(file_path, skiprows=3, header=None)
            df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df = df.drop_duplicates().ffill()
            df = preprocess(df)
            df.to_csv(PROCESSED / f"{coin}_features.csv")
        except Exception as e:
            print(f"Error processing {coin}: {e}")

if __name__ == "__main__":
    run()