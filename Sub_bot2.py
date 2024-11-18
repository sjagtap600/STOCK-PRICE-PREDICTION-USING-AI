import os
import yfinance as yf
import discord
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import pandas as pd
import numpy as np

top_stock_companies = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'FB', 'BRK-B', 'SPY', 
          'BABA', 'JPM', 'WMT', 'V', 'T', 'UNH', 'PFE', 'INTC', 'VZ', 'ORCL']

async def send_daily_trade_updates_plot(top_stock_company, existing_channel):
    top_stock_company_df = yf.download(
        top_stock_company, period="1d", interval="1m")

    top_stock_company_df.plot(y='Close', linewidth=0.85)

    plt.xlabel('Datetime')
    plt.ylabel('Close')
    plt.title('Latest stock prices of {company}'.format(
        company=top_stock_company))
    plt.savefig('images/daily_trade_updates_plot_1.png')

    top_stock_company_df.plot(
        y=['Open', 'High', 'Low', 'Close', 'Adj Close'], linewidth=0.85)

    plt.xlabel('Datetime')
    plt.ylabel('Value')
    plt.title('Latest stock prices of {company}'.format(
        company=top_stock_company))
    plt.savefig('images/daily_trade_updates_plot_2.png')

    my_files = [
        discord.File('images/daily_trade_updates_plot_1.png'),
        discord.File('images/daily_trade_updates_plot_2.png')
    ]

    await existing_channel.send('Latest stock prices:', files=my_files)

    os.remove('images/daily_trade_updates_plot_1.png')
    os.remove('images/daily_trade_updates_plot_2.png')


async def send_history_plot(stock_companies, existing_channel):
    df = yf.download(stock_companies[0])
    ax = df.plot(y='Close', label=stock_companies[0], linewidth=0.85)

    for stock_company in stock_companies:
        if stock_company != stock_companies[0]:
            df = yf.download(stock_company)
            df.plot(ax=ax, y='Close', label=stock_company, linewidth=0.85)

    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title("Historical stock details")

    plt.savefig('images/history.png')

    await existing_channel.send(file=discord.File('images/history.png'))

    os.remove('images/history.png')


async def send_history_plot_in_date_interval(args, existing_channel):
    length = len(args)

    try:
        date_obj1 = datetime.datetime.strptime(args[length-1], '%Y-%m-%d')
        date_obj2 = datetime.datetime.strptime(args[length-2], '%Y-%m-%d')
    except ValueError:
        await existing_channel.send("Incorrect data format, should be YYYY-MM-DD")
        return

    arr = []
    for i in range(length-2):
        arr.append(args[i])
    if set(tuple(arr)).issubset(tuple(top_stock_companies)):
        df = yf.download(arr[0], start=args[length-2], end=args[length-1])
        ax = df.plot(y='Close', label=arr[0], linewidth=0.85)

        for stock_company in arr:
            if stock_company != arr[0]:
                df = yf.download(
                    stock_company, start=args[length-2], end=args[length-1])
                df.plot(ax=ax, y='Close', label=stock_company, linewidth=0.85)

        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.title("Historical stock details for {date1} - {date2}".format(
            date1=args[length-2], date2=args[length-1]))

        plt.savefig('images/history_date_interval.png')

        await existing_channel.send(file=discord.File('images/history_date_interval.png'))

        os.remove('images/history_date_interval.png')
    else:
        await existing_channel.send("Invalid set of companies.")


# New Functions for AI-based Stock Recommendations
def predict_stock_prices(symbol, start_date, end_date, N=10):
    """
    Predicts stock prices using linear regression.
    """
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if stock_data.empty:
        return {"error": f"No data found for {symbol}"}

    data = stock_data['Close'].dropna().reset_index(drop=True)

    X, y = [], []
    for i in range(len(data) - N):
        X.append(data[i:i + N].values)
        y.append(data[i + N])

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    latest_data = data[-N:].values.reshape(1, -1)
    next_day_price = model.predict(latest_data)[0]

    return {
        "symbol": symbol,
        "next_day_price": next_day_price,
        "test_rmse": test_rmse
    }


async def recommend_stocks(existing_channel, start_date="2023-01-01", end_date="2023-12-31", top_n=5):
    """
    Recommends top stocks to invest in based on predicted returns.
    """
    recommendations = []

    for symbol in top_stock_companies:
        try:
            result = predict_stock_prices(symbol, start_date, end_date)
            if "error" not in result:
                recommendations.append({
                    "symbol": result["symbol"],
                    "next_day_price": result["next_day_price"],
                    "test_rmse": result["test_rmse"]
                })
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    recommendations.sort(key=lambda x: x["next_day_price"], reverse=True)

    message = "**Top Stock Recommendations**\n"
    for i, rec in enumerate(recommendations[:top_n]):
        message += f"{i+1}. {rec['symbol']}: Predicted Price = ${rec['next_day_price']:.2f}, RMSE = {rec['test_rmse']:.2f}\n"

    await existing_channel.send(message)


# New Risk Management Function
def calculate_risk_management(symbol, start_date, end_date, max_loss=0.05, max_gain=0.10):
    """
    This function will calculate the risk management parameters based on a given stock symbol.
    It will suggest whether the stock is within an acceptable risk range.
    max_loss: maximum percentage loss you're willing to accept.
    max_gain: maximum percentage gain you want to secure.
    """
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    if stock_data.empty:
        return {"error": f"No data found for {symbol}"}

    # Calculate the percentage change from the start price to the end price
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    
    percentage_change = (end_price - start_price) / start_price

    # Determine risk status
    if percentage_change <= -max_loss:
        risk_status = "High Risk (Potential loss exceeds max loss)"
    elif percentage_change >= max_gain:
        risk_status = "Good Gain (Target gain reached)"
    else:
        risk_status = "Moderate Risk (Within acceptable range)"

    return {
        "symbol": symbol,
        "start_price": start_price,
        "end_price": end_price,
        "percentage_change": percentage_change,
        "risk_status": risk_status
    }
