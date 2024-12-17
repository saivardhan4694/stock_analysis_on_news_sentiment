# Importing Libraries
import yfinance as yf
from finvizfinance.quote import finvizfinance
import pandas as pd
from newsapi import NewsApiClient
import datetime

# Initialize the News API Client
NEWS_API_KEY = 'a8d19eac3c234d339f3478a6ea312b58' 

class ProcessData:
    def __init__(self):
        pass
        import streamlit as st

    def fetch_news_last_30_days(self, ticker):
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=30)
        
        # Fetch articles from the last 30 days
        all_articles = newsapi.get_everything(
            q=ticker,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=today.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=100  # Maximum allowed per request
        )

        # Create a DataFrame for the news articles
        if all_articles['status'] == 'ok':
            articles_df = pd.DataFrame([
                {'Title': article['title'], 'Link': article['url'], 'Published At': article['publishedAt']}
                for article in all_articles['articles']
            ])
            return articles_df
        else:
            return pd.DataFrame()

    # Function to get and process news data
    def get_news_data(self, ticker, classify_sentiment):
        # Fetch articles using NewsAPI
        news_df = self.fetch_news_last_30_days(ticker)

        # Check if the DataFrame is empty
        if news_df.empty:
            return pd.DataFrame(columns=['Title', 'sentiment', 'Date', 'DateOnly'])

        # Preprocess: Convert 'Title' to lowercase
        news_df['Title'] = news_df['Title'].str.lower()

        # Add 'Date' column from 'Published At'
        news_df['Date'] = pd.to_datetime(news_df['Published At'])
        news_df['DateOnly'] = news_df['Date'].dt.date

        # Apply sentiment classification
        news_df['sentiment'] = news_df['Title'].apply(classify_sentiment)

        # Postprocess: Convert sentiment to uppercase and filter out 'NEUTRAL'
        news_df['sentiment'] = news_df['sentiment'].str.upper()
        news_df = news_df[news_df['sentiment'] != 'NEUTRAL']

        # Select relevant columns and return
        return news_df[['Title', 'sentiment', 'Date', 'DateOnly']]


    # Function to group and process sentiment data
    def process_sentiment_data(self, news_df, window = 10):

        # Reshape data to have df with columns: Date, # of positive Articles, # of negative Articles
        grouped = news_df.groupby(['DateOnly', 'sentiment']).size().unstack(fill_value=0)
        grouped = grouped.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)

        # Create rolling averages that count number of positive and negative sentiment articles within past t days
        grouped['day_avg_positive'] = grouped['POSITIVE'].rolling(window=window, min_periods=1).sum()
        grouped['day_avg_negative'] = grouped['NEGATIVE'].rolling(window=window, min_periods=1).sum()

        # Create "Percent Positive" by creating percentage measure
        grouped['day_pct_positive'] = grouped['POSITIVE'] / (grouped['POSITIVE'] + grouped['NEGATIVE'])
        result_df = grouped.reset_index()

        return result_df
    
    
    # Function to fetch and process stock data
    def get_stock_data(self, ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date) # Pull ticker data
        stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100 # Transform closing value to percent change in closing value since previous day 
        return stock_data

    # Function to combine sentiment and stock data
    def combine_data(self, result_df, stock_data):
        combined_df = result_df.set_index('DateOnly').join(stock_data[['Pct_Change']], how='inner')
        combined_df['lagged_day_pct_positive'] = combined_df['day_pct_positive'].shift(1) # Lag sentiment feature by 1 day for temporal alignment
        return combined_df

    # Function to calculate Pearson correlation
    def calculate_correlation(self, combined_df):
        correlation_pct_change = combined_df[['lagged_day_pct_positive', 'Pct_Change']].corr().iloc[0, 1]
        return correlation_pct_change