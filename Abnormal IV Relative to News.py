import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# List of stocks to analyze
stocks_to_analyze = []
while True:
    stock = input("Enter the stock ticker you would like to analyze (or type 'done' to finish): ")
    if stock.lower() == 'done':
        break
    stocks_to_analyze.append(stock.upper())


api_key = input("Enter your NewsAPI key (https://newsapi.org/): ")   # Replace with your actual NewsAPI key (https://newsapi.org/)

if api_key == '':
    break



class FocusedStockAnalyzer:
    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)

    def get_news_coverage(self, tickers):

        # Fetch news coverage for each stock
        print("Fetching news coverage...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Define news sources and financial keywords for search queries
        sources = 'the-wall-street-journal,the-new-york-times,cnbc,financial-times,reuters'
        financial_keywords = ['stock', 'shares', 'earnings', 'investors', 'market']
        
        # Map ticker symbols to their respective long names
        stock_names = {ticker: yf.Ticker(ticker).info.get('longName', ticker) for ticker in tickers}

        news_counts = {}
        for name in stock_names:
            total_articles = 0
            
            try:
                # Construct query for ticker
                ticker_query = f'"{stock_names[name]}" AND ({" OR ".join(financial_keywords)})'
                print(f"Searching for: {stock_names[name]}")
                
                # Fetch news articles
                ticker_articles = self.newsapi.get_everything(
                    q=ticker_query,
                    #sources=sources,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy'
                )
                
                print(f"Found {ticker_articles['totalResults']} articles for ticker {name}")
                total_articles += ticker_articles['totalResults']
                print(f"Total of {total_articles} articles found for {stock_names[name]}")
            except Exception as e:
                print(f"Error fetching news for {stock_names[name]}: {str(e)}")
            
            news_counts[name] = total_articles
        
        print(f"Final news counts: {news_counts}")
        
        # Apply logarithmic transformation to news counts
        log_news_counts = {ticker: math.log(count + 1) for ticker, count in news_counts.items()}
        print(f"Log-transformed news counts: {log_news_counts}")
        
        # Ask user if they want to use log-transformed news counts
        decision = input("\nLog(1) or Standard news counts(2)?: Enter 1 or 2: ")
        print('\n')
        if decision == '1':
            return log_news_counts
        else:
            return news_counts

    def calculate_abnormal_iv(self, ticker):
        print(f"Calculating AbnormalIV for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            now = datetime.now()
            
            # Fetch option expiration dates
            expiration_dates = stock.options
            if not expiration_dates:
                print(f"No options data available for {ticker}")
                return None

            days_to_exp = [(datetime.strptime(date, '%Y-%m-%d') - now).days for date in expiration_dates]
            
            # Fetch implied volatility for ATM call options
            ivs = []
            for exp_date in expiration_dates:
                try:
                    options = stock.option_chain(exp_date)
                    
                    current_price = self.get_stock_price(ticker)
                    if current_price is None:
                        raise ValueError("Unable to get current stock price")
                    
                    # Try to get ATM call, if not available, get the first OTM call
                    atm_calls = options.calls[options.calls['strike'] >= current_price]
                    if not atm_calls.empty:
                        atm_call = atm_calls.iloc[0]
                    else:
                        atm_call = options.calls.iloc[-1]  # Get the last ITM call if no OTM available
                    
                    ivs.append(atm_call['impliedVolatility'])
                except Exception as e:
                    print(f"Error processing options for {ticker} expiring on {exp_date}: {str(e)}")
                    continue
            
            # Check if we have enough data to calculate AbnormalIV
            if len(days_to_exp) < 2 or len(ivs) < 2:
                print(f"Insufficient data to calculate AbnormalIV for {ticker}")
                return None
            
            # Interpolate IV for 30 and 60 days to expiration
            f = interpolate.interp1d(days_to_exp, ivs, kind='linear', fill_value='extrapolate')
            iv_30 = f(30)
            iv_60 = f(60)
            
            # Calculate Abnormal IV
            abnormal_iv = (iv_30 - iv_60) / (1/30 - 1/60)
            return abnormal_iv
        except Exception as e:
            print(f"Error calculating AbnormalIV for {ticker}: {str(e)}")
            return None

    def get_stock_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            if history.empty:
                print(f"No recent price data available for {ticker}")
                return None
            return history['Close'].iloc[-1]
        except Exception as e:
            print(f"Error getting stock price for {ticker}: {str(e)}")
            return None

    def get_market_cap(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('marketCap', None)
        except Exception as e:
            print(f"Error getting market cap for {ticker}: {str(e)}")
            return None

    def analyze_stocks(self, tickers):
        print("Analyzing stocks...")
        news_coverage = self.get_news_coverage(tickers)
        results = []
        
        # Calculate abnormal IV and other metrics for each stock
        for ticker in tickers:
            try:
                abnormal_iv = self.calculate_abnormal_iv(ticker)
                price = self.get_stock_price(ticker)
                market_cap = self.get_market_cap(ticker)
                news_coverage_mkt_weight = abnormal_iv / market_cap if market_cap is not None and abnormal_iv is not None else None
                
                if price is not None:  # Only add to results if we have a valid price
                    results.append({
                        'ticker': ticker,
                        'news_coverage': news_coverage.get(ticker, 0),
                        'abnormal_iv': abnormal_iv,
                        'price': price,
                        'market_cap': market_cap,
                        'news_coverage_mkt_weight': news_coverage_mkt_weight
                    })
                else:
                    print(f"Skipping {ticker} due to missing price data")
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        return results





def visualize_results(df):
    # Select only numeric columns for correlation analysis and visualization
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Correlation Heatmap", "News Coverage by Market Cap vs AbnormalIV"),
                        specs=[[{"type": "heatmap"}, {"type": "scatter"}]])
    
    # Correlation heatmap
    corr_matrix = numeric_df.corr()
    heatmap = go.Heatmap(z=corr_matrix.values,
                         x=corr_matrix.columns,
                         y=corr_matrix.columns,
                         colorscale='RdBu_r',  # Note the '_r' to reverse the color scale
                         zmin=-1, zmax=1,
                         text=corr_matrix.values,
                         texttemplate='%{text:.2f}',
                         textfont={"size":10})
    fig.add_trace(heatmap, row=1, col=1)
    
    # Scatter plot: News Coverage vs AbnormalIV
    scatter = go.Scatter(x=df['news_coverage_mkt_weight'],
                         y=df['abnormal_iv'],
                         mode='markers+text',
                         marker=dict(size=df['market_cap']/1e9,  # Size based on market cap (in billions)
                                     sizemode='area',
                                     sizeref=2.*max(df['market_cap']/1e9)/(40.**2),
                                     sizemin=4),
                         text=df['ticker'],
                         textposition="top center",
                         hoverinfo='text',
                         hovertext=[f"{row['ticker']}<br>News Coverage: {row['news_coverage_mkt_weight']:.2f}<br>Abnormal IV: {row['abnormal_iv']:.2f}<br>Market Cap: ${row['market_cap']/1e9:.2f}B" for _, row in df.iterrows()])
    fig.add_trace(scatter, row=1, col=2)
    
    # Update layout
    fig.update_layout(height=600, width=1200, title_text="Stock Analysis Visualization")
    fig.update_xaxes(title_text="News Coverage by Market Cap", row=1, col=2)
    fig.update_yaxes(title_text="Abnormal IV", row=1, col=2)
    
    # Show the plot
    fig.show()

def main():
    analyzer = FocusedStockAnalyzer(api_key)
    
    results = analyzer.analyze_stocks(stocks_to_analyze)
    
    if not results:
        print("No valid data to analyze. Please check your stock list and try again.")
        return

    # Convert results to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Normalize the news coverage by market cap weight
    df_results['news_coverage_mkt_weight'] = (
        df_results['news_coverage_mkt_weight'] - df_results['news_coverage_mkt_weight'].min()) / (
        df_results['news_coverage_mkt_weight'].max() - df_results['news_coverage_mkt_weight'].min())
    
    print("\nResults:")
    print(df_results)
    
    if not df_results.empty:
        # Select only numeric columns for correlation analysis
        numeric_columns = df_results.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_results[numeric_columns].corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Identify stocks with high news coverage and high AbnormalIV
        if 'news_coverage' in df_results.columns and 'abnormal_iv' in df_results.columns:
            median_news = df_results['news_coverage_mkt_weight'].median()
            median_iv = df_results['abnormal_iv'].median()
            high_attention_stocks = df_results[
                (df_results['news_coverage_mkt_weight'] >= median_news) & 
                (df_results['abnormal_iv'] >= median_iv)
            ]
            print("\nStocks with high news coverage and high AbnormalIV:")
            print(high_attention_stocks[['ticker', 'news_coverage', 'abnormal_iv']])
            print(f"\nMedian news coverage: {median_news}")
            print(f"Median abnormal IV: {median_iv}")
        else:
            print("\nUnable to identify high attention stocks due to missing data.")
        
        # Visualize results
        visualize_results(df_results)
    else:
        print("No data available for analysis after processing.")

if api_key == 'enter-your-api-key':
    print("Please enter your actual NewsAPI (https://newsapi.org/) key in the 'api_key' variable.")
else:
    if __name__ == "__main__":
        main()
