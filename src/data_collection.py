import yfinance as yf
from pathlib import Path
from config import COINS, START_DATE
import matplotlib.pyplot as plt

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

def fetch_data():
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
    fetch_data()
    plot_close("BTC-USD")
    plot_close("ETH-USD")
    plot_close("BNB-USD")
    plot_close("SOL-USD")
    plot_close("XRP-USD")