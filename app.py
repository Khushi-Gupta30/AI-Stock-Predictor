import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta

# =========================
# LOAD MODEL
# =========================

model = load_model('Stock Predictions Model.keras')

# =========================
# UI
# =========================

st.title("📈 AI Stock Market Predictor")

st.sidebar.title("📊 Stock Selection")

stocks = {

    # US Stocks
    "Google (GOOG)" : "GOOG",
    "Apple (AAPL)" : "AAPL",
    "Microsoft (MSFT)" : "MSFT",
    "Amazon (AMZN)" : "AMZN",
    "Tesla (TSLA)" : "TSLA",
    "NVIDIA (NVDA)" : "NVDA",
    "Meta (META)" : "META",
    "Netflix (NFLX)" : "NFLX",
    "AMD (AMD)" : "AMD",
    "Intel (INTC)" : "INTC",

    # Indian Stocks
    "Reliance (RELIANCE.NS)" : "RELIANCE.NS",
    "TCS (TCS.NS)" : "TCS.NS",
    "Infosys (INFY.NS)" : "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)" : "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)" : "ICICIBANK.NS",
    "SBI (SBIN.NS)" : "SBIN.NS",
    "Wipro (WIPRO.NS)" : "WIPRO.NS",
    "Tata Motors (TATAMOTORS.NS)" : "TATAMOTORS.NS",
    "Bharti Airtel (BHARTIARTL.NS)" : "BHARTIARTL.NS",
    "ITC (ITC.NS)" : "ITC.NS",

    # Crypto
    "Bitcoin (BTC-USD)" : "BTC-USD",
    "Ethereum (ETH-USD)" : "ETH-USD",
    "Solana (SOL-USD)" : "SOL-USD",

    # Indices
    "NIFTY 50 (^NSEI)" : "^NSEI",
    "SENSEX (^BSESN)" : "^BSESN",
    "NASDAQ (^IXIC)" : "^IXIC",
    "S&P 500 (^GSPC)" : "^GSPC"
}

input_method = st.sidebar.radio(
    "Choose Input Method",
    [
        "Popular Stocks",
        "Custom Symbol"
    ]
)

if input_method == "Popular Stocks":

    selected_company = st.sidebar.selectbox(
        "Select Stock",
        list(stocks.keys())
    )

    stock = stocks[selected_company]

else:

    stock = st.sidebar.text_input(
        "Enter Stock Symbol",
        "GOOG"
    )

st.sidebar.success(
    f"Selected: {stock}"
)
# =========================
# DOWNLOAD DATA
# =========================

start = "2015-01-01"
end = date.today().strftime("%Y-%m-%d")

data = yf.download(
    stock,
    start=start,
    end=end
)

if len(data) == 0:
    st.error("Invalid stock symbol")
    st.stop()

st.subheader("Stock Data")
st.write(data.tail())

# =========================
# MOVING AVERAGES
# =========================

ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

# MA50

st.subheader("Price vs MA50")

fig1 = plt.figure(figsize=(10,6))
plt.plot(data.Close,label="Close Price")
plt.plot(ma50,label="MA50")
plt.legend()
st.pyplot(fig1)

# MA50 + MA100

st.subheader("Price vs MA50 vs MA100")

fig2 = plt.figure(figsize=(10,6))
plt.plot(data.Close,label="Close Price")
plt.plot(ma50,label="MA50")
plt.plot(ma100,label="MA100")
plt.legend()
st.pyplot(fig2)

# MA100 + MA200

st.subheader("Price vs MA100 vs MA200")

fig3 = plt.figure(figsize=(10,6))
plt.plot(data.Close,label="Close Price")
plt.plot(ma100,label="MA100")
plt.plot(ma200,label="MA200")
plt.legend()
st.pyplot(fig3)

# =========================
# TRAIN TEST SPLIT
# =========================

data_train = pd.DataFrame(
    data['Close'][0:int(len(data)*0.80)]
)

data_test = pd.DataFrame(
    data['Close'][int(len(data)*0.80):]
)

# =========================
# SCALING
# =========================

scaler = MinMaxScaler(
    feature_range=(0,1)
)

data_train_scaled = scaler.fit_transform(
    data_train
)

past_100_days = data_train.tail(100)

data_test_combined = pd.concat(
    [past_100_days,data_test],
    ignore_index=True
)

data_test_scaled = scaler.transform(
    data_test_combined
)

# =========================
# CREATE TEST SEQUENCES
# =========================

x_test = []
y_test = []

for i in range(
    100,
    data_test_scaled.shape[0]
):

    x_test.append(
        data_test_scaled[i-100:i]
    )

    y_test.append(
        data_test_scaled[i,0]
    )

x_test = np.array(x_test)
y_test = np.array(y_test)

# =========================
# PREDICT TEST DATA
# =========================

y_predict = model.predict(x_test)

y_predict = scaler.inverse_transform(
    y_predict
)

y_test = y_test.reshape(-1,1)

y_test = scaler.inverse_transform(
    y_test
)

# =========================
# ACTUAL VS PREDICTED
# =========================

st.subheader(
    "Actual Price vs Predicted Price"
)

fig4 = plt.figure(figsize=(10,6))

plt.plot(
    y_test,
    'b',
    label='Actual Price'
)

plt.plot(
    y_predict,
    'r',
    label='Predicted Price'
)

plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

st.pyplot(fig4)

# =========================
# TOMORROW PREDICTION
# =========================

last_100_days = data['Close'].tail(100)

last_100_days_scaled = scaler.transform(
    last_100_days.values.reshape(-1,1)
)

X_future = np.array(
    [last_100_days_scaled]
)

tomorrow_prediction = model.predict(
    X_future,
    verbose=0
)

tomorrow_prediction = scaler.inverse_transform(
    tomorrow_prediction
)

st.subheader(
    "Tomorrow Predicted Closing Price"
)

st.success(
    f"{tomorrow_prediction[0][0]:.2f}"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Selected Stock",
        stock
    )

with col2:
    st.metric(
        "Tomorrow Prediction",
        f"{tomorrow_prediction[0][0]:.2f}"
    )

# =========================
# FUTURE FORECAST FUNCTION
# =========================

def forecast_days(days):

    batch = last_100_days_scaled.copy()

    future_predictions = []

    for _ in range(days):

        x_input = batch.reshape(
            1,
            100,
            1
        )

        pred = model.predict(
            x_input,
            verbose=0
        )

        future_predictions.append(
            pred[0,0]
        )

        batch = np.vstack(
            [
                batch[1:],
                pred
            ]
        )

    future_predictions = np.array(
        future_predictions
    ).reshape(-1,1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    )

    future_dates = []

    last_date = data.index[-1]

    for i in range(
        1,
        days+1
    ):

        future_dates.append(
            last_date + timedelta(days=i)
        )

    forecast_df = pd.DataFrame(
        {
            "Date":future_dates,
            "Predicted Price":
            future_predictions.flatten()
        }
    )

    return forecast_df

# =========================
# 7 DAY FORECAST
# =========================

forecast_7 = forecast_days(7)

st.subheader(
    "7 Day Forecast"
)

st.dataframe(forecast_7)

fig5 = plt.figure(figsize=(10,6))

plt.plot(
    forecast_7["Date"],
    forecast_7["Predicted Price"]
)

plt.title(
    "Next 7 Day Forecast"
)

plt.xticks(rotation=45)

st.pyplot(fig5)

# =========================
# 30 DAY FORECAST
# =========================

forecast_30 = forecast_days(30)

st.subheader(
    "30 Day Forecast"
)

st.dataframe(forecast_30)

fig6 = plt.figure(figsize=(10,6))

plt.plot(
    forecast_30["Date"],
    forecast_30["Predicted Price"]
)

plt.title(
    "Next 30 Day Forecast"
)

plt.xticks(rotation=45)

st.pyplot(fig6)