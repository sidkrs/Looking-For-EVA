# Looking For EVA
# Stock Analyzer: News Coverage and Abnormal Implied Volatility

## Overview

This program is a stock analysis tool that examines the relationship between news coverage and abnormal implied volatility (AbnormalIV) for a set of stocks. It's inspired by research on retail investor behavior around earnings announcements and aims to identify stocks that might be subject to increased retail investor attention and potential mispricing.

## Purpose

The main purpose of this tool is to:

1. Quantify news coverage for selected stocks
2. Calculate abnormal implied volatility (AbnormalIV)
3. Analyze the relationship between news coverage and AbnormalIV
4. Identify stocks that may be prone to retail investor overreaction

## Concept: Expected Announcement Volatility (EAV)

This program applies the concept of Expected Announcement Volatility (EAV) from academic research on retail investor behavior. 

Link to paper: https://tinyurl.com/ea-vol


Key points:

- EAV is approximated using AbnormalIV, calculated from option prices
- Higher AbnormalIV suggests the market expects more volatility around upcoming events (e.g., earnings announcements)
- Stocks with high EAV tend to attract more retail investor attention, especially when combined with increased news coverage

## Features

1. **News Coverage Analysis**: Fetches and quantifies recent news articles for each stock using the NewsAPI.
2. **AbnormalIV Calculation**: Computes abnormal implied volatility using option chain data from yfinance.
3. **Market Cap Weighting**: Adjusts news coverage by market capitalization to account for size differences.
4. **Correlation Analysis**: Examines relationships between news coverage, AbnormalIV, and other stock metrics.
5. **Visualization**: Generates heatmaps and scatter plots to visualize relationships between variables.

## How It Works

1. Fetches news data and calculates news coverage for each stock
2. Retrieves option data and calculates AbnormalIV
3. Collects additional stock data (price, market cap)
4. Analyzes correlations between variables
5. Identifies stocks with both high news coverage and high AbnormalIV
6. Visualizes results for easy interpretation

## Potential Applications

- Identifying stocks that may be subject to retail investor overreaction
- Exploring potential mispricing in options markets
- Studying the relationship between media attention and market expectations
- Developing trading strategies based on news coverage and implied volatility patterns

## Limitations and Considerations

- Limited to stocks within the S&P 500
- Relies on the accuracy and timeliness of news and financial data sources

*Ultimately, if there is high vol and high media coverage --> sell vol

## Future Application
Build an options strategy that sells higher than normal vol (Iron Condor)
  
