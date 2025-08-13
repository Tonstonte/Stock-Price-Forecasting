# 📈 Stock Price Prediction Dashboard

A comprehensive Streamlit web application for stock price analysis and forecasting using Linear Regression and ARIMA models. This application provides real-time stock data analysis with interactive visualizations and predictive modeling capabilities.

## 🚀 Features

### Linear Regression Stock Analysis
- **Multi-stock support**: Analyze popular stocks (AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA)
- **Technical indicators**: Uses moving averages, volatility, and volume metrics
- **Performance metrics**: R² score, MSE, RMSE evaluation
- **Future predictions**: Forecast stock prices up to 60 days ahead
- **Interactive charts**: Historical data, model predictions, and forecasts

### ARIMA S&P 500 Forecasting
- **Time series analysis**: Advanced ARIMA modeling for S&P 500 index
- **Configurable parameters**: Multiple ARIMA order options
- **Statistical validation**: Stationarity tests and model diagnostics
- **Confidence intervals**: 95% confidence bands around forecasts
- **Residual analysis**: Model performance evaluation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection (for fetching stock data)

### Quick Setup

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd stock-prediction-dashboard
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas numpy yfinance matplotlib seaborn scikit-learn statsmodels plotly
   ```

   Or using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📦 Dependencies

- **streamlit** (>=1.28.0) - Web application framework
- **pandas** (>=1.5.0) - Data manipulation and analysis
- **numpy** (>=1.24.0) - Numerical computing
- **yfinance** (>=0.2.0) - Yahoo Finance API for stock data
- **matplotlib** (>=3.6.0) - Basic plotting
- **seaborn** (>=0.12.0) - Statistical visualization
- **scikit-learn** (>=1.3.0) - Machine learning algorithms
- **statsmodels** (>=0.14.0) - Statistical modeling
- **plotly** (>=5.15.0) - Interactive plotting

## 🎯 Usage

### Getting Started
1. Launch the application using `streamlit run app.py`
2. Select your desired model from the dropdown menu
3. Configure parameters using the sidebar controls
4. Click the analysis button to generate predictions

### Linear Regression Analysis
1. **Select Model**: Choose "Linear Regression - Stock Analysis"
2. **Pick Stock**: Select from available stock tickers
3. **Set Parameters**: 
   - Historical period (1y, 2y, 5y, max)
   - Forecast days (5-60 days)
4. **Run Analysis**: Click "Run Linear Regression Analysis"
5. **View Results**: 
   - Model performance metrics
   - Interactive prediction charts
   - Feature importance analysis

### ARIMA Forecasting
1. **Select Model**: Choose "ARIMA - S&P 500 Forecasting"
2. **Configure Settings**:
   - Historical period (2y, 5y, 10y, max)
   - Forecast days (10-90 days)
   - ARIMA order parameters
3. **Run Forecast**: Click "Run ARIMA Forecast"
4. **Analyze Results**:
   - Model diagnostics (AIC, BIC)
   - Forecast with confidence intervals
   - Residual analysis charts

## 📊 Model Details

### Linear Regression Features
- **Day Index**: Sequential day numbering
- **Moving Averages**: 5-day and 20-day moving averages
- **Volatility**: 20-day rolling standard deviation
- **Volume**: 20-day average trading volume
- **Price Changes**: Daily percentage changes

### ARIMA Configuration
- **Orders Available**: (1,1,1), (2,1,2), (1,1,2), (2,1,1)
- **Stationarity Testing**: Augmented Dickey-Fuller test
- **Model Selection**: AIC/BIC criteria
- **Validation**: Residual analysis and diagnostics

## 🎨 User Interface

### Main Dashboard
- Clean, intuitive layout with sidebar navigation
- Model selection dropdown with clear descriptions
- Real-time parameter adjustment
- Professional chart visualizations

### Interactive Charts
- **Plotly Integration**: Zoom, pan, hover functionality
- **Multiple Series**: Historical, predicted, and forecast data
- **Confidence Bands**: Statistical uncertainty visualization
- **Responsive Design**: Works on desktop and mobile

## ⚠️ Important Notes

### Data Source
- Stock data is fetched in real-time from Yahoo Finance
- Historical data availability depends on Yahoo Finance API
- Market hours and holidays may affect data availability

### Model Limitations
- **Linear Regression**: Best for trend analysis, may not capture complex patterns
- **ARIMA**: Assumes stationarity, works well for time series with clear patterns
- **Predictions**: Not financial advice - for educational purposes only

### Performance Considerations
- Large historical periods may take longer to process
- ARIMA model fitting can be computationally intensive
- Internet connection required for data fetching

## 🚨 Disclaimer

**This application is for educational and research purposes only. The predictions and analysis provided should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- More sophisticated models (LSTM, Prophet)
- Portfolio optimization features
- Enhanced visualization options
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**App won't start:**
- Check Python version (3.8+ required)
- Ensure all dependencies are installed
- Try upgrading Streamlit: `pip install --upgrade streamlit`

**Data fetching errors:**
- Check internet connection
- Yahoo Finance may have temporary outages
- Try different stock symbols or time periods

**Model errors:**
- Insufficient data: Choose longer historical periods
- ARIMA convergence: Try different order parameters
- Feature calculation: Some stocks may lack volume data

**Performance issues:**
- Reduce historical data period
- Close other applications to free memory
- Try different model parameters

### Getting Help
- Check the Streamlit documentation: https://docs.streamlit.io
- Yahoo Finance API: https://pypi.org/project/yfinance/
- Scikit-learn documentation: https://scikit-learn.org/
- Statsmodels documentation: https://www.statsmodels.org/

---

**Version**: 1.0.0  
**Last Updated**: August 2025  
**Python Version**: 3.8+

