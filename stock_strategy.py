import numpy as np
import pandas as pd
import yfinance as yf

print("--------------------------------------------------")
print(" QUANTITATIVE STOCK BACKTESTER (MA CROSSOVER)")
print("--------------------------------------------------")

# Configuration
ticker = "SPY"  # You can change this to any ticker (e.g., "AAPL", "QQQ", "NVDA")
start_date = "2005-01-01"
end_date = "2009-01-01"

try:
  print(f"1. Downloading historical data for {ticker} from Yahoo Finance...")
  df = yf.download(ticker, start=start_date, end=end_date, progress=False)

  # Handle modern yfinance multi-index column structures if present
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  if df.empty:
    print(f"❌ Error: No data found for ticker {ticker}.")
    input("\nPress Enter to exit...")
    exit()

  print(f"   -> Success! Loaded {len(df):,} trading days of data.")
except Exception as e:
  print(f"❌ Error downloading data: {e}")
  input("\nPress Enter to exit...")
  exit()

print("2. Calculating Moving Averages and Trading Signals...")

# Define Moving Average Windows
short_window = 50
long_window = 200

df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()

# Strategy Signal: 1 when 50-day SMA > 200-day SMA (Bullish), 0 when below (Cash)
df["Signal"] = 0
df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1

# Calculate Returns
df["Market_Return"] = df["Close"].pct_change()
# Shift signal by 1 day to avoid look-ahead bias (trade executed on next day's open/close)
df["Strategy_Return"] = df["Signal"].shift(1) * df["Market_Return"]

df = df.dropna()

# Compute Cumulative Performance
market_cum = (1 + df["Market_Return"]).prod() - 1
strategy_cum = (1 + df["Strategy_Return"]).prod() - 1

# Calculate Annualized Volatility & Sharpe Ratio approximation
market_vol = df["Market_Return"].std() * np.sqrt(252)
strategy_vol = df["Strategy_Return"].std() * np.sqrt(252)

print("\n==================================")
print(f"   BACKTEST RESULTS: {ticker}")
print("==================================")
print(
    f"Period Evaluated:       {df.index.min().strftime('%Y-%m-%d')} to"
    f" {df.index.max().strftime('%Y-%m-%d')}"
)
print(f"Benchmark (Buy & Hold): {market_cum * 100:.2f}% (Vol: {market_vol*100:.1f}%)")
print(
    f"Strategy (MA Crossover):{strategy_cum * 100:.2f}% (Vol:"
    f" {strategy_vol*100:.1f}%)"
)
print("==================================")

input("\nPress Enter to close the program...")
