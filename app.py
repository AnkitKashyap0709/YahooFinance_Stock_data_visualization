# For deploying this project online using streamlit:
# 1. Add this python file in a github repository.
# 2. Create a streamlit account and add the project's repository link in the streamlit app when creating a "new project".
# 3. Customize the streamlit link and deploy.


import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from plotly import graph_objs as go

#Ticker Resources
START_DATE = "2015-01-01"
TGT_DATE = datetime.strftime((datetime.now() - timedelta(1)), "%Y-%m-%d")

_POSS_TICKERS = ["F", "AMZN", "TSLA", "INTC", "BAC", "VALE", "AMD", "AAPL", "SOFI", "KVUE", "GOOG", 
    "NVDA", "PLTR", "AAL", "CMCSA", "NIO", "T", "GOOG", "PFE", "META", "MSFT", "GM", "AVTR", "CCL", "GOLD",
    "NU", "RIVN", "SNAP", "CVX", "BBD", "VZ", "GRAB", "NOK", "XOM", "PLUG", "NYCB", "UAA", "SQ", "ENPH", "AES", 
    "RTX", "NEM", "AGNC", "PYPL", "PBR", "LU", "UBER", "BMY", "PCG", "BCS"
]


# Add description as it is done already below, inside the below Python dictionary for each company according to the company's symbol. This will be in use when we need to view the "Description of a company" in the app.
TICKER_INFOER = {
    "F":{
        "Name": "FaceBook",
        "Description": "FaceBook is an online social network"
    },
    "GOOG": {
        "Name": "Google",
        "Description": "Google is a search engine"
    },
}


#region: FUnc

#Cache Problems Streamlit
def load_data(ticker):
    try:
        data = yf.download(str(ticker).strip().lower(), START_DATE, TGT_DATE)
        data.reset_index(inplace=True)
        return data
    except Exception:
        return str(f"> Cannot load ({ticker}) try clearing cache (by pressing \"c\") and update the selected tickers.")

def ask_tickers(tickers):
    _TICKERS = {}
    for x in tickers:
        dat = load_data(x)
        if isinstance(dat, str):
            st.text(dat)
        elif "Date" not in dat.columns or "Adj Close" not in dat.columns:
            st.text(f"Invalid dataframe columns: {dat.columns}")
        else:
            _TICKERS[x] = dat
            print(dat.tail(5))
    return _TICKERS

def set_stoc(k, ticker):
    ticker["Stock"] = k
    return ticker

def rem_stoc(col, ticker):
    return ticker.drop(col, axis=1)

#Prediction Code
def prediq(tickers_df, period):
    _TICKERS = {}
    _PLOTZ = {}
    for k, v in tickers_df.items():
        v = v.rename(columns={"Date": "ds", "Adj Close": "y"})
        model = Prophet()
        model.fit(v)
        future = model.make_future_dataframe(periods=period)
            
        forecast = model.predict(future)
        _PLOTZ[k] = plot_plotly(model, forecast)
            
        forecast = forecast.rename(columns={"ds": "Date", "yhat": "Adj Close"})
        _TICKERS[k] = forecast
            
    return _TICKERS, _PLOTZ

# For showing dynamic legend information of the plot (Open, Close) inside the plot when predicting the chart
def cmb_tickers(tickers_df, fig):
    _DF = pd.DataFrame()
    for k, v in tickers_df.items():
        if "Date" not in v.columns or "Adj Close" not in v.columns:
            st.text(f"Invalid dataframe columns: {v.columns}")
            st.stop()
        v = set_stoc(k, v)
        fig.add_trace(go.Scatter(x = v["Date"], y = v["Adj Close"], name = k))
        fig.layout.update(title_text='Time Series data with Range slider', xaxis_rangeslider_visible=True)
        _DF = pd.concat([_DF, v])
    return _DF
        
#View Chart 

# For showing dynamic legend information of the plot (Open, Close) inside the plot when viewing the chart
def cmb_openclose(tickers_df, fig):
    _DF = pd.DataFrame()
    for k, v in tickers_df.items():
        if "Date" not in v.columns or "Adj Close" not in v.columns:
            st.text(f"Invalid dataframe columns: {v.columns}")
            st.stop()
        v = set_stoc(k, v)
        
        # Adding traces for Open and Close prices
        fig.add_trace(go.Scatter(x=v["Date"], y=v["Open"], name=f"{k}_open"))
        fig.add_trace(go.Scatter(x=v["Date"], y=v["Close"], name=f"{k}_close"))   

        fig.layout.update(title_text = 'Time Series data with Range slider', xaxis_rangeslider_visible = True)
        _DF = pd.concat([_DF, v])
    return _DF

def plot_graph(fig):
    # Cross-hair showing the details of the open and close of a stock at a particular point (line-guide)
    fig.update_xaxes(showspikes = True, spikecolor = "grey", spikemode = "across", spikesnap = "cursor", spikethickness = 1)
    fig.update_yaxes(showspikes = True, spikecolor = "grey", spikemode = "across", spikethickness = 1)
    fig.update_layout(hovermode = "x")
    return fig

#endregion: FUnc


# Streamlit app starts here
def main():
    st.title('Yahoo Finance Data Analysis')

    # Selection of what the user wants to do
    option = st.selectbox(
        "What would you like to do?",
        ("View Chart", "Predict Company", "Description of Company")
    )

    if option == "View Chart":
        st.subheader("View Chart")
        _OPT_TICKERS = st.multiselect("Select data region for chart", _POSS_TICKERS)
        if len(_OPT_TICKERS) > 0:
            _TCKRS = ask_tickers(_OPT_TICKERS)
            fig = go.Figure()
            df = cmb_openclose(_TCKRS, fig)
            st.plotly_chart(plot_graph(fig), use_container_width = True)
        else:
            st.warning("Please select at least one dataset.")

    elif option == "Predict Company":
        st.subheader("Predict Company")
        _OPT_TICKERS = st.multiselect("Select datasets for prediction", _POSS_TICKERS)
        if len(_OPT_TICKERS) > 0:
            _TCKRS = ask_tickers(_OPT_TICKERS)
            n_years = st.slider("Years of prediction: ", 1, 5)
            period = n_years * 365
            dff, pltrz = prediq(_TCKRS, period)
            fig = go.Figure()
            df = cmb_tickers(dff, fig)
            
            st.plotly_chart(plot_graph(fig))

            for k, v in pltrz.items():
                st.subheader(f"Charts for: {k}")
                st.plotly_chart(plot_graph(v))
        else:
            st.warning("Please select at least one dataset.")

    elif option == "Description of Company":
        st.subheader("Description of Company")
        _OPT_TICKERS = st.multiselect("Select datasets for description", _POSS_TICKERS)
        if len(_OPT_TICKERS) > 0:
            for k in _OPT_TICKERS:
                dat = TICKER_INFOER.get(k, {})
                st.header(f"{dat.get('Name', 'Unknown')} ({k})")
                st.text(dat.get("Description", "No Description Available"))
        else:
            st.warning("Please select at least one dataset.")

if __name__ == '__main__':
    main()
