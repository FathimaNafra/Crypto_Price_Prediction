import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import COINS, TEST_SIZE

DATA_PATH = Path("data/processed")

for coin in COINS:
    print(f"Training baseline models for {coin}...")
    df = pd.read_csv(DATA_PATH / f"{coin}_features.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # Define target
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    # Features & labels
    x = df.drop(columns=["Date", "Close", "Target"])
    y = df["Target"]

    # Train-test split
    split_index = int((1 - TEST_SIZE) * len(df))
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    print(f"\n{coin} Results:")
    print(f"Linear Regression - MAE: {lr_mae:.4f}, RMSE: {lr_rmse:.4f}")
    print(f"Random Forest - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}")
models={
    "LinearRegression":LinearRegression(),
    "RandomForest":RandomForestRegressor(n_estimators=100,random_state=42)  

}

#Train & Evaluate
for name,model in models.items():
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)
    mae=mean_absolute_error(y_test,predictions)
    rmse=np.sqrt(mean_squared_error(y_test,predictions))
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

