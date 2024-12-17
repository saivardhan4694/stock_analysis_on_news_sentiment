from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import streamlit as st
import holidays
from prophet import Prophet
import pandas as pd
from pathlib import Path

class ModelTrainer:
    def __init__(self):
        # load the model and tokenizer
        self.model_path = Path(__file__).resolve().parent / "model_files"
        self.model = TFDistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)

    # Function to classify sentiment
    def classify_sentiment(self, title):

        # Preprocess and tokenize the title
        encodings = self.tokenizer(title, truncation=True, padding=True, max_length=128, return_tensors='tf')

        # Make prediction
        predictions = self.model.predict(encodings).logits
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Decode the predicted class to sentiment label
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_label = label_map[predicted_class]

        return predicted_label

    # Function to get future dates excluding weekends and holidays
    def get_future_dates(self, start_date, num_days):
        us_holidays = holidays.US()
        future_dates = []
        current_date = start_date
        while len(future_dates) < num_days:
            if current_date.weekday() < 5 and current_date not in us_holidays:
                future_dates.append(current_date)
            current_date += pd.Timedelta(days=1)
        return future_dates

    def fit_and_forecast(self, combined_df, forecast_steps=3):
        # Drop NaN values and align the indices of endog and exog
        combined_df.dropna(inplace=True)
        endog = combined_df['Pct_Change'].dropna()
        exog = combined_df['lagged_day_pct_positive'].dropna()

        # Align indices to ensure both variables have the same length
        aligned_index = endog.index.intersection(exog.index)
        endog = endog.loc[aligned_index]
        exog = exog.loc[aligned_index]

        # Ensure exog is 2D, but endog remains 1D
        exog = exog.values.reshape(-1, 1)

        # Should now be (n_samples, 1)

        # Fit the ARIMAX model
        model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
        fit = model.fit(disp=False)

        # Forecasting steps
        future_dates = self.get_future_dates(combined_df.index[-1], forecast_steps)
        
        # Prepare future exogenous values; ensure they have the correct shape
        future_exog = combined_df['lagged_day_pct_positive'][-forecast_steps:].values.reshape(-1, 1)
        
        # Get forecast
        forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        return forecast_mean, forecast_ci, future_dates


    

    def forecast_with_prophet(self, stock_data, forecast_days):
        forecasts = {}

        # Columns to forecast
        features = ['Close', 'High', 'Low', 'Open', 'Volume']

        for feature in features:
            # Prepare the data for Prophet
            df = stock_data[['Date', feature]].copy()
            df.columns = ['ds', 'y']  # Prophet expects columns 'ds' (date) and 'y' (value)

            # Initialize and fit the Prophet model
            model = Prophet()
            model.fit(df)

            # Create a dataframe for future dates
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Forecast the future values
            forecast = model.predict(future)

            # Store the forecast in the dictionary
            forecasts[feature] = forecast

        return model, forecasts


    # Function to create and display plot
    def create_plot(self, combined_df, forecast_mean, forecast_ci, forecast_index):
        # Standardize the sentiment proportion
        sentiment_std = (combined_df['day_pct_positive'] - combined_df['day_pct_positive'].mean()) / combined_df['day_pct_positive'].std()

        fig = go.Figure()
        
        # Add standardized sentiment proportion
        fig.add_trace(go.Scatter(
            x=combined_df.index, 
            y=sentiment_std, 
            name='Standardized Sentiment Proportion', 
            line=dict(color='blue'), 
            mode='lines'
        ))
        
        # Add stock percentage change
        fig.add_trace(go.Scatter(
            x=combined_df.index, 
            y=combined_df['Pct_Change'], 
            name='Stock Pct Change', 
            line=dict(color='green'), 
            yaxis='y2', 
            mode='lines'
        ))
        
        # Add forecasted stock percentage change
        fig.add_trace(go.Scatter(
            x=forecast_index, 
            y=forecast_mean, 
            name='Forecasted Pct Change', 
            line=dict(color='red'), 
            mode='lines'
        ))
        
        # Add confidence intervals for the forecast
        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_index, forecast_index[::-1]]),
            y=np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Update layout with appropriate y-axis ranges
        fig.update_layout(
            title='Sentiment Proportion and Stock Percentage Change with Forecast',
            xaxis_title='Date',
            yaxis=dict(
                title='Standardized Sentiment Proportion',
                titlefont=dict(color='blue')
            ),
            yaxis2=dict(
                title='Stock Pct Change',
                titlefont=dict(color='green'),
                overlaying='y',
                side='right'
            ),
            template='plotly_dark'
        )
        st.plotly_chart(fig)