import pandas as pd
from pathlib import Path
from config import COINS

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
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
        df = pd.read_csv(RAW / f"{coin}.csv", skiprows=3, header=None)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.drop_duplicates().ffill()
        df = preprocess(df)
        df.to_csv(PROCESSED / f"{coin}_features.csv")

if __name__ == "__main__":
    run()