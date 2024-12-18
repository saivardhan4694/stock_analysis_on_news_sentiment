import streamlit as st
from model_trainer import ModelTrainer
from data_processing import ProcessData
import pandas as pd
from datetime import datetime
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from finvizfinance.quote import finvizfinance


# ----------------------------
# Function to fetch stock data
# ----------------------------
@st.cache_data
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - datetime.timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

# -----------------------------------
# Function to fetch real-time prices
# -----------------------------------
def fetch_realtime_prices(tickers):
    """
    Fetch real-time prices for a list of tickers.
    """
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = {
            'Current Price': stock.history(period='1d')['Close'].iloc[-1],
            'Previous Close': stock.history(period='2d')['Close'].iloc[0],
        }
    return pd.DataFrame(data).T

# ---------------------------
# Function to fetch news data
# ---------------------------
def fetch_news_data(ticker):
    """
    Fetch the latest news articles for the given ticker.
    """
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    news_df = news_df.tail(7)
    return news_df[['Title', 'Link']]

# Sidebar Title
st.sidebar.title("Predicting Stock Prices by News Sentiment")

# User Input for Stock Ticker
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL):", value='AAPL')

# Current date
today = datetime.date.today()

# Enforce that the start date is no earlier than one month ago
min_start_date = today - datetime.timedelta(days=30)

# User Input for Start Date
user_start_date = st.sidebar.date_input(
    "Start Date", 
    value=min_start_date,  # Default to 1 month ago
    min_value=min_start_date,  # Earliest allowed date
    max_value=today  # Latest allowed date
)

end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Enforce that forecast_days cannot be below 7
forecast_days = st.sidebar.number_input(
    "Enter number of days to forecast:", 
    min_value=7,  # Minimum value allowed
    max_value=365, 
    value=7,  # Default value
    key="forecast_days_input"
)


# ---------------------------
# Sidebar: Real-time Stock Display
# ---------------------------
st.sidebar.title("Interactive Stock Data Explorer")

# Sidebar section for real-time stock prices of selected symbols
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        last_price = real_time_data['Close'].iloc[-1]
        change = last_price - real_time_data['Open'].iloc[0]
        pct_change = (change / real_time_data['Open'].iloc[0]) * 100
        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")


# Create Tabs
tab1, tab2 = st.tabs(["Stock Data Explorer", "Stock Price Predictions"])

# ------------------------------
# Tab 1: Stock Data Explorer
# ------------------------------

with tab1:
    # ------------------------------
    # Main App: Stock Data Explorer
    # ------------------------------
    st.title(f"Stock Data Analysis for {ticker}")

    # Date range picker
    
    stock_data = fetch_stock_data(ticker, user_start_date, end_date)

    # Display the fetched data
    st.subheader("Fetched Stock Data")
    st.dataframe(stock_data, use_container_width=True, height=400)

    # ------------------------------
    # Custom Column Selection
    # ------------------------------
    st.sidebar.subheader("Select Columns to Display")
    columns_to_display = st.sidebar.multiselect(
        "Choose Columns",
        stock_data.columns.tolist(),
        default=['Date', 'Close', 'Volume']
    )

    # Display selected columns
    st.write("### Selected Data View")
    st.dataframe(stock_data[columns_to_display], use_container_width=True)

    # ------------------------------
    # Chart Options
    # ------------------------------
    st.sidebar.subheader("Chart Options")
    chart_type = st.sidebar.selectbox("Select Chart Type", ['Line', 'Bar', 'Area'])
    selected_y_axis = st.sidebar.multiselect(
        "Select Y-Axis Metrics",
        ['Close', 'Open', 'High', 'Low', 'Volume'],
        default=['Close']
    )


    # Plot based on user selection
    st.subheader(f"{chart_type} Chart for Selected Metrics")
    if chart_type == 'Line':
        fig = px.line(stock_data, x='Date', y=selected_y_axis, title=f"{chart_type} Chart of Selected Metrics")
    elif chart_type == 'Bar':
        fig = px.bar(stock_data, x='Date', y=selected_y_axis, title=f"{chart_type} Chart of Selected Metrics")
    elif chart_type == 'Area':
        fig = px.area(stock_data, x='Date', y=selected_y_axis, title=f"{chart_type} Chart of Selected Metrics")

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Toggle Technical Indicators
    # ------------------------------
    st.sidebar.subheader("Technical Indicators")
    show_moving_averages = st.sidebar.checkbox("Show Moving Averages", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI", value=False)
    show_volatility = st.sidebar.checkbox("Show Volatility", value=False)

    # Display news articles for the selected stock
    st.sidebar.subheader(f"Latest News for {ticker}")
    news_data = fetch_news_data(ticker)
    for index, row in news_data.iterrows():
        st.sidebar.write(f"[{row['Title']}]({row['Link']})")

    if show_moving_averages:
        st.subheader("Moving Averages")
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Close Price'))
        ma_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'].rolling(window=5).mean(), name='5-Day MA'))
        ma_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'].rolling(window=10).mean(), name='10-Day MA'))
        ma_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'].rolling(window=30).mean(), name='30-Day MA'))
        ma_fig.update_layout(title="Moving Averages", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(ma_fig, use_container_width=True)

    if show_rsi:
        st.subheader("Relative Strength Index (RSI)")
        stock_data['RSI'] = stock_data['Close'].diff().fillna(0).rolling(window=14).mean()
        rsi_fig = px.line(stock_data, x='Date', y='RSI', title="RSI Over Time")
        st.plotly_chart(rsi_fig, use_container_width=True)

    if show_volatility:
        st.subheader("Volatility")
        stock_data['Volatility'] = stock_data['Close'].rolling(window=5).std()
        vol_fig = px.line(stock_data, x='Date', y='Volatility', title="5-Day Volatility")
        st.plotly_chart(vol_fig, use_container_width=True)

    # ------------------------------
    # Correlation Matrix
    # ------------------------------
    st.subheader("Correlation Matrix")
    correlation = stock_data[['Close', 'Open', 'High', 'Low', 'Volume']].corr()
    st.dataframe(correlation.style.background_gradient(cmap='coolwarm'))

    # Footer
    st.write("### Stock Data Explorer")
    st.write("Explore stock data, visualize trends, and analyze technical indicators.")
    


# ------------------------------
# Tab 2: Stock Price Predictions
# ------------------------------
with tab2:
    st.header("Stock Price Change Analysis and Forecast")

    # Run Analysis Button
    run_button = st.button("Run Pct Change Analysis")

    if run_button:
        # Initialize ModelTrainer and ProcessData objects
        model_trainer_obj = ModelTrainer()
        process_data_obj = ProcessData()

        # fetch and process data
        news_df = process_data_obj.get_news_data(ticker, model_trainer_obj.classify_sentiment)
        result_df = process_data_obj.process_sentiment_data(news_df, window=forecast_days)
        start_date = user_start_date.strftime('%Y-%m-%d')
        end_date = result_df['DateOnly'].max().strftime('%Y-%m-%d')
        stock_data = process_data_obj.get_stock_data(ticker, start_date, end_date)
        combined_df = process_data_obj.combine_data(result_df, stock_data)
        
        # caluculate corelations
        correlation_pct_change = process_data_obj.calculate_correlation(combined_df)
        st.write(f'Pearson correlation between lagged sentiment score and stock percentage change: {correlation_pct_change}')

        # run model
        forecast_mean, forecast_ci, forecast_index = model_trainer_obj.fit_and_forecast(combined_df)
        model_trainer_obj.create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)

    # Fetch stock data and run forecast when a checkbox is selected
    run_forecast = st.checkbox("Run Prophet Forecast along with pct_change forecast.")

    if run_forecast:
        # Initialize ModelTrainer and ProcessData objects
        model_trainer_obj2 = ModelTrainer()
        process_data_obj2 = ProcessData()
        # Fetch stock data
        start_date = user_start_date.strftime('%Y-%m-%d')
        forecast_end_date = datetime.date.today().strftime('%Y-%m-%d')
        stock_data_for_forecast = process_data_obj2.get_stock_data(ticker, start_date, end_date)

        # Ensure the stock data has a 'Date' column
        stock_data_for_forecast.reset_index(inplace=True)
        stock_data['Date'] = stock_data['Date'].dt.date

        # Run the Prophet forecast function
        model, forecasts = model_trainer_obj2.forecast_with_prophet(stock_data_for_forecast, forecast_days)

        # Create a dropdown selector for the user to choose a feature
        selected_feature = st.selectbox(
            "Select a feature to display the forecast:", 
            options=list(forecasts.keys()),
            key="prophet_feature_selector"  # Unique key to maintain state
        )

        # get data for the selected feature
        st.write(f"### {selected_feature} Forecast fot next {forecast_days} days")
        selected_df = forecasts[selected_feature]

        # Display the styled DataFrame in Streamlit
        future_forecasts_df = selected_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        styled_df = future_forecasts_df.style \
            .highlight_max(color='lightgreen', axis=0) \
            .highlight_min(color='red', axis=0)
        st.dataframe(styled_df)


        # Plot the forecast for the selected feature
        st.write(f"Forcast plot for {selected_feature}")
        fig = plot_plotly(model, selected_df)
        fig.update_layout(
            title=f"Forecast Plot for {selected_feature}",
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            template='plotly' 
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"Components Plot for {selected_feature}")
        st.plotly_chart(plot_components_plotly(model, selected_df))

st.write("By Bhavana Naidu & Sai Vardhan.M")
