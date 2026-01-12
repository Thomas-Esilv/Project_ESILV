import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta


# --- Functions ---

def fecth_data_yahoo(ticker : str, period : str):
    data = yf.download(tickers=ticker,period=period,group_by="ticker")
    data = data.droplevel(level=0,axis=1)
    return data

def metrics(data):
    last_price = data.Close.iloc[-1]
    previous_close = data.Close.iloc[0]
    pct_change = (last_price - previous_close) / previous_close * 100
    max_price = data.High.max()
    min_price = data.Low.min()
    volume = int(data.Volume.mean())

    return last_price, pct_change, max_price, min_price, volume 

def add_technical_indicators(data):
    data['MA10'] = ta.ma("ema",data.Close,lenght=10)
    data['MA30'] = ta.ma("ema",data.Close,lenght=30)
    data['MA50'] = ta.ma("ema",data.Close,lenght=50)
    return data

# --- SIDEBAR ---

st.sidebar.title("Dashboard")

ticker_dict = {"CAC40":"^FCHI","EuroStoxx 50":"^STOXX50E","Dow Jones":"^DJI","S&P500":"^GSPC"}
key_ticker = st.sidebar.selectbox("Ticker",options=["CAC40","EuroStoxx 50","Dow Jones","S&P500"])
ticker = ticker_dict[key_ticker]

period = st.sidebar.selectbox("Time Period",
                                options =["1d","1wk","1mo","3mo","6mo","1y","ytd","max"])

chart_type = st.sidebar.selectbox("Chart Type", options=["Candlestick","Line"])

indicators = st.sidebar.multiselect("Technical Indicators",
                                    options = ["MA10","MA30","MA50"],
                                    default=None)

update = st.sidebar.button("Update")

# --- Main Area ---

if update:

    data = fecth_data_yahoo(ticker=ticker, period=period)
    data = add_technical_indicators(data=data)
    last_price, pct_change ,max_price, min_price, volume = metrics(data=data)


    st.subheader(f"{key_ticker} / {ticker}")

    #Display main metrics

    st.metric("Last Price",value=f"{last_price:.2f}", delta=f"{pct_change:.2f} %")

    left,center,right = st.columns(3)
    left.metric("Highest Price",value=f"{max_price:.2f}")
    center.metric("Lowest Price", value=f"{min_price:.2f}")
    right.metric("Average Volume",value=f"{volume:,}")

    # Creation of the stock price chart
    if chart_type == 'Candlestick':
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data.Open,
                                             high=data.High,
                                             low=data.Low,
                                             close=data.Close,
                                             name="OHLC")])

    else:
        fig = px.line(x=data.index,y=data.Close)

    # Add selected technical indicators to the chart

    for indicator in indicators:
        if indicator == "MA10":
            fig.add_trace(go.Scatter(x=data.index,
                                     y=data.MA10,
                                     name="Moving Average 10 days"))
        elif indicator == "MA30":
            fig.add_trace(go.Scatter(x=data.index,
                                     y=data.MA30,
                                     name="Moving Average 30 days"))
        elif indicator == "MA50":
            fig.add_trace(go.Scatter(x=data.index,
                                     y=data.MA50,
                                     name="Moving Average 50 days"))

    
    # Format graph
    fig.update_layout(title = f"{key_ticker} {period} Chart",
                    xaxis_title = "Time",
                    yaxis_title = 'Points',
                    height = 600)

    # Plot the graph
    st.plotly_chart(fig, use_container_width=True)

    # Display historical data
    st.subheader("Historical Data")
    st.dataframe(data=data)





