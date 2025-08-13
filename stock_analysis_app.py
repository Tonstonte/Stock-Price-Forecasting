import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Analysis & Forecasting Dashboard")
st.markdown("Choose between Linear Regression for stock analysis or ARIMA forecasting for S&P 500")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Analysis Type:",
    ["Linear Regression - Stock Analysis", "ARIMA - S&P 500 Forecasting"]
)

def fetch_stock_data(symbol, period="2y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def prepare_features_for_regression(data):
    """Prepare features for linear regression"""
    df = data.copy()
    df['Day'] = range(len(df))
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    
    # Remove NaN values
    df = df.dropna()
    return df

def linear_regression_analysis():
    st.header("🔍 Linear Regression Stock Analysis")
    
    # Stock selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_symbol = st.selectbox(
            "Select Stock:",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        )
    
    with col2:
        period = st.selectbox(
            "Time Period:",
            ["1y", "2y", "5y", "max"],
            index=1
        )
    
    with col3:
        forecast_days = st.slider("Forecast Days:", 5, 60, 30)
    
    if st.button("Run Linear Regression Analysis", type="primary"):
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            data = fetch_stock_data(stock_symbol, period)
            
            if data is not None and len(data) > 50:
                # Prepare features
                df = prepare_features_for_regression(data)
                
                # Feature selection
                features = ['Day', 'MA_5', 'MA_20', 'Volatility', 'Volume_MA']
                X = df[features]
                y = df['Close']
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("MSE", f"{mse:.2f}")
                with col3:
                    st.metric("RMSE", f"{np.sqrt(mse):.2f}")
                
                # Create forecast
                last_day = df['Day'].iloc[-1]
                future_days = range(last_day + 1, last_day + forecast_days + 1)
                
                # Extend moving averages and other features for forecast
                last_values = df.tail(20)
                future_data = []
                
                for day in future_days:
                    # Simple linear extrapolation for moving averages
                    ma_5_trend = (last_values['MA_5'].iloc[-1] - last_values['MA_5'].iloc[-5]) / 5
                    ma_20_trend = (last_values['MA_20'].iloc[-1] - last_values['MA_20'].iloc[-10]) / 10
                    
                    future_ma_5 = last_values['MA_5'].iloc[-1] + ma_5_trend * (day - last_day)
                    future_ma_20 = last_values['MA_20'].iloc[-1] + ma_20_trend * (day - last_day)
                    
                    future_data.append({
                        'Day': day,
                        'MA_5': future_ma_5,
                        'MA_20': future_ma_20,
                        'Volatility': last_values['Volatility'].mean(),
                        'Volume_MA': last_values['Volume_MA'].mean()
                    })
                
                future_df = pd.DataFrame(future_data)
                future_predictions = model.predict(future_df[features])
                
                # Plotting
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Test predictions
                test_dates = X_test.index
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=y_pred,
                    mode='lines',
                    name='Model Predictions',
                    line=dict(color='orange', dash='dash')
                ))
                
                # Future predictions
                future_dates = pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color='red', dash='dot'),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f'{stock_symbol} Stock Price Prediction - Linear Regression',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_
                })
                
                fig_importance = px.bar(
                    importance_df,
                    x='Feature',
                    y='Coefficient',
                    title='Linear Regression Coefficients'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
            else:
                st.error("Unable to fetch sufficient data for analysis")

def arima_forecasting():
    st.header("📊 ARIMA S&P 500 Forecasting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period = st.selectbox(
            "Historical Period:",
            ["2y", "5y", "10y", "max"],
            index=1,
            key="arima_period"
        )
    
    with col2:
        forecast_days = st.slider("Forecast Days:", 10, 90, 30, key="arima_forecast")
    
    with col3:
        arima_order = st.selectbox(
            "ARIMA Order (p,d,q):",
            ["(1,1,1)", "(2,1,2)", "(1,1,2)", "(2,1,1)"],
            index=0
        )
    
    if st.button("Run ARIMA Forecast", type="primary"):
        with st.spinner("Fetching S&P 500 data and running ARIMA model..."):
            # Fetch S&P 500 data
            data = fetch_stock_data("^GSPC", period)
            
            if data is not None and len(data) > 100:
                # Prepare data
                prices = data['Close'].dropna()
                
                # Check stationarity
                adf_result = adfuller(prices)
                is_stationary = adf_result[1] < 0.05
                
                # Display data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(prices))
                with col2:
                    st.metric("ADF p-value", f"{adf_result[1]:.4f}")
                with col3:
                    st.metric("Stationary", "Yes" if is_stationary else "No")
                
                # Parse ARIMA order
                order = tuple(map(int, arima_order.strip("()").split(",")))
                
                try:
                    # Fit ARIMA model
                    model = ARIMA(prices, order=order)
                    fitted_model = model.fit()
                    
                    # Make forecast
                    forecast = fitted_model.forecast(steps=forecast_days)
                    forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()
                    
                    # Model summary
                    st.subheader("Model Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("AIC", f"{fitted_model.aic:.2f}")
                    with col2:
                        st.metric("BIC", f"{fitted_model.bic:.2f}")
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Historical data (last 200 points for better visualization)
                    hist_data = prices.tail(200)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data.values,
                        mode='lines',
                        name='Historical S&P 500',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    future_dates = pd.date_range(
                        start=prices.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast,
                        mode='lines+markers',
                        name='ARIMA Forecast',
                        line=dict(color='red', dash='dot'),
                        marker=dict(size=6)
                    ))
                    
                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast_ci.iloc[:, 0],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast_ci.iloc[:, 1],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='95% Confidence Interval',
                        fillcolor='rgba(255,0,0,0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f'S&P 500 ARIMA{order} Forecast - {forecast_days} Days',
                        xaxis_title='Date',
                        yaxis_title='Index Value',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residual analysis
                    st.subheader("Residual Analysis")
                    residuals = fitted_model.resid
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            x=residuals.index,
                            y=residuals,
                            mode='lines',
                            name='Residuals',
                            line=dict(color='green')
                        ))
                        fig_resid.update_layout(
                            title='Residuals Over Time',
                            xaxis_title='Date',
                            yaxis_title='Residuals'
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)
                    
                    with col2:
                        fig_hist = px.histogram(
                            x=residuals,
                            nbins=30,
                            title='Residuals Distribution'
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Forecast summary table
                    st.subheader("Forecast Summary")
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower CI': forecast_ci.iloc[:, 0],
                        'Upper CI': forecast_ci.iloc[:, 1]
                    })
                    
                    st.dataframe(forecast_df.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error fitting ARIMA model: {e}")
                    st.info("Try different ARIMA parameters or check if the data needs more preprocessing.")
                    
            else:
                st.error("Unable to fetch sufficient S&P 500 data for ARIMA analysis")

# Main application logic
def main():
    if model_choice == "Linear Regression - Stock Analysis":
        linear_regression_analysis()
    else:
        arima_forecasting()

if __name__ == "__main__":
    main()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** This is for educational purposes only. Not financial advice.")
