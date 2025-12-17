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
    """
    Add technical indicators and features for price prediction
    """
    # Price-based features
    df["Daily_Return"] = df["Close"].pct_change()
    df["Price_Change"] = df["Close"].diff()
    
    # Moving averages
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()
    
    # Exponential moving average
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    
    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]
    
    # Volatility indicators
    df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
    df["Volatility_14"] = df["Daily_Return"].rolling(14).std()
    df["Volatility_30"] = df["Daily_Return"].rolling(30).std()
    
    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Mid"] - (bb_std * 2)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    
    # Price position within Bollinger Bands
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
    # Volume-based (if available)
    if "Volume" in df.columns:
        df["Volume_MA_7"] = df["Volume"].rolling(7).mean()
    
    # Remove rows with NaN values from feature engineering
    return df.dropna()


def run():
    for coin in COINS:
        print(f"Processing {coin}...")
        file_path = RAW / f"{coin}.csv"
        if not file_path.exists():
            print(f"✗ Warning: {file_path} does not exist. Skipping {coin}.")
            continue
        try:
            # Read CSV, skip the metadata rows (keep first row as header)
            df = pd.read_csv(file_path, skiprows=[1, 2])
            
            # Find the Date column
            if "Date" in df.columns:
                date_col = "Date"
            else:
                date_col = df.columns[0]
            
            # Set Date as index
            df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=[date_col])
            df.set_index(date_col, inplace=True)
            df.index.name = "Date"
            
            # Keep only Close price column and convert to numeric
            if "Close" in df.columns:
                df = df[["Close"]].copy()
                df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
            else:
                print(f"✗ Error processing {coin}: Available columns: {df.columns.tolist()}")
                continue
            
            # Remove NaN values and duplicates
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            # Apply preprocessing
            df = preprocess(df)
            
            # Save processed features
            df.to_csv(PROCESSED / f"{coin}_features.csv")
            print(f"✓ Processed {coin}: {len(df)} records with {len(df.columns)} features")
        except Exception as e:
            import traceback
            print(f"✗ Error processing {coin}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run()