import yfinance as yf
import pandas as pd
from datetime import datetime

def download_stock_data(ticker, start="2015-01-01", end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ["Date"] + [f"{ticker}_{col}" for col in df.columns[1:]]
    return df

def load_macro_data():
    # Placeholder for GDP/Inflation data, replace with actual load method
    df = pd.read_csv("data/gdp_inflation.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_sentiment_data():
    # Placeholder for sentiment data, ensure it includes 'Date' column
    df = pd.read_csv("data/vedl_news.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def merge_all():
    vedl = download_stock_data("VEDL.NS")
    nifty50 = download_stock_data("^NSEI")
    nifty_metal = download_stock_data("^CNXMETAL")
    gdp_infl = load_macro_data()
    sentiment = load_sentiment_data()

    # Merge all on 'Date'
    df = vedl.merge(nifty50, on="Date", how="left")
    df = df.merge(nifty_metal, on="Date", how="left")
    df = df.merge(gdp_infl, on="Date", how="left")
    df = df.merge(sentiment, on="Date", how="left")

    df.dropna(inplace=True)  # Optionally handle missing
    df.to_csv("data/final_merged_data.csv", index=False)
    print("Merged data saved to data/final_merged_data.csv")

if __name__ == "__main__":
    merge_all()