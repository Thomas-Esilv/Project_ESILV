import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime as dt
import backtrader as bt
import numpy as np

### Functions

def get_yahoo_data(list_tickers: list, start: dt.datetime, end: dt.datetime, interval: str = "1d"):
    """
    Download data from the yahoo finance API and return a dataframe
    """
    try:
        data = yf.download(tickers=list_tickers, start=start, end=end, 
                           interval=interval, group_by="ticker" ,auto_adjust=True)
    except Exception as e:
        return f"Error while importing data from yahoo : {e}"
    else:
        return data
    
def run_strategy(strategy_class, data, name, params, 
                 cash_start = 1000000.0, commission_broker = 0.001,
                 slippage_pct = 0.001) -> dict:
    """
    Run the backtest for the strategy
    """
    
    cerebro = bt.Cerebro()
    
    if params is None:
        cerebro.addstrategy(strategy_class)
    else: 
        cerebro.addstrategy(strategy_class, parameters = params)

    for elt in data:
        cerebro.adddata(elt)
    
    cerebro.broker.setcash(cash_start)
    cerebro.broker.setcommission(commission=commission_broker) 
    cerebro.broker.set_slippage_perc(perc=slippage_pct) 

    # cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.addsizer(bt.sizers.AllInSizerInt)
    #cerebro.addsizer(bt.sizers.PercentSizer, percents = 30)

    cerebro.addanalyzer(bt.analyzers.Returns, _name = "returns")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='periodstats')
    cerebro.addanalyzer(bt.analyzers.PyFolio)
    
    results = cerebro.run()
    strat = results[0]

    rtot_log =  strat.analyzers.returns.get_analysis().get("rtot")
    
    return {
        "name": name,
        "start_value" : cerebro.broker.startingcash,
        "final_value": cerebro.broker.getvalue(),
        "total_return_pct": (np.exp(rtot_log)-1)*100,
        "annualized_return_pct": strat.analyzers.returns.get_analysis().get("rnorm100"),
        "sharpe": strat.analyzers.sharpe.get_analysis().get('sharperatio', None),
        "max_drawdown": strat.analyzers.drawdown.get_analysis().max.drawdown,
        "total_trades": strat.analyzers.trades.get_analysis().total.total,
        "number_positive_years" : strat.analyzers.periodstats.get_analysis().get("positive"),
        "number_negative_years" : strat.analyzers.periodstats.get_analysis().get("negative"),
        "number_flat_years" : strat.analyzers.periodstats.get_analysis().get("nochange"),
        "best_year_perf_pct" : strat.analyzers.periodstats.get_analysis().get("best") * 100,
        "worst_year_perf_pct" : strat.analyzers.periodstats.get_analysis().get("worst") * 100,
        "average_annual_return_pct" : strat.analyzers.periodstats.get_analysis().get("average") * 100,
        "annual_std_pct" : strat.analyzers.periodstats.get_analysis().get("stddev") * 100,
        "pyfolio" : strat.analyzers.getbyname("pyfolio") 
    }

def compute_index_backtrader(data: list, weights: dict):
    res = 0
    for d in data:
        asset_name = d._name
        if asset_name in weights:
            res+= d.close / d.close.array[0] * weights[asset_name] 
    return res

def compute_index(data: list, weights: dict, underlyings: list) -> pd.DataFrame:
    normalized_data = {}
    for s in underlyings:
        normalized_data[s] = data[s]["Close"] / data[s]["Close"].iloc[0]
    
    res = np.zeros(data.shape[0])
    for s in underlyings:
        res+= weights[s] * normalized_data[s]
    res_df = pd.DataFrame(index=data.index,
                       data=res.tolist(),
                       columns=["index"])
    return res_df

### Backtrader Strategies

class BuyAndHold(bt.Strategy):
    """
    Buy & Hold Strategy Class
    """
    params = (
        ("parameters",{"weights": {'Cash':1.0}}),
    )

    def __init__(self):
        params_dict = self.p.parameters
        self.current_weights = params_dict.get("weights",{'Cash':1.0})

    def rebalance_portfolio(self):
        for asset, target_weight in self.current_weights.items():
            if asset != "Cash":
                self.order_target_percent(data=asset ,target=target_weight)

    def notify_order(self, order):

        # if the order fais, we reduce the target weight by 1%
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            asset_name = order.data._name
            if asset_name in self.current_weights:
                self.current_weights[asset_name] = max(0, self.current_weights[asset_name]-0.01)
        
        self.order = None 

    def next(self):
        if not self.position:
            self.rebalance_portfolio()

class MeanRevertion(bt.Strategy):
    """
    Mean Revertion Strategy Class
    """
    params = (
        ("parameters",{"weights": {'Cash':1.0}, 
                       "std_lower_boundary": 2.0, 
                       "std_upper_boundary": 2.0,
                       "std_mean_boundary":0.01, 
                       "period_mean_reversion": 20}),
    )

    def __init__(self):
        self.params_dict = self.p.parameters
        self.current_weights = self.params_dict.get("weights",{'Cash':1.0})
        self.index = compute_index_backtrader(self.datas, weights=self.current_weights)
        self.sma = bt.indicators.MovingAverageSimple(self.index,
                                                     period=self.params_dict.get("period_mean_reversion",20))
        self.std = bt.indicators.StandardDeviation(self.index,
                                                    period=self.params_dict.get("period_mean_reversion",20))

    def rebalance_portfolio(self,positif_sign = 1):
        for asset, target_weight in self.current_weights.items():
            if asset != "Cash":
                if positif_sign == 1:
                    self.order_target_percent(data=asset ,target=target_weight)
                elif positif_sign == -1:
                    self.order_target_percent(data=asset ,target=-target_weight)
                else:
                    self.close(data=asset)

    def notify_order(self, order):

        # if the order fais, we reduce the target weight by 1%
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            asset_name = order.data._name
            if asset_name in self.current_weights:
                self.current_weights[asset_name] = max(0, self.current_weights[asset_name]-0.01)
        
        self.order = None 

    def next(self):
        
        if not self.position:
            # If the index is under the lower boundary, we buy
            if self.index[0] <= self.sma[0] - self.std[0] * self.params_dict.get("std_lower_boundary",2.0):
                self.rebalance_portfolio(positif_sign=1)

            # If the index is over the upper boundary, we sell
            elif self.index[0] >= self.sma[0] + self.std[0] * self.params_dict.get("std_upper_boundary",2.0):
                self.rebalance_portfolio(positif_sign=-1)
            
        else:
            # If the index is close to the mean, we close the position
            if abs(self.index[0] - self.sma[0]) <= self.std[0] * self.params_dict.get("std_mean_boundary",0.01) :
                self.rebalance_portfolio(positif_sign=0)
            
class Momentum(bt.Strategy):
    """
    Momentum Strategy Class
    """
    params = (
        ("parameters",{"weights": {'Cash':1.0}, 
                       "short_moving_average": 30,
                       "long_moving_average": 200, 
                       "rebalancing_period_momentum": 5,
                       }),
    )

    def __init__(self):
        self.params_dict = self.p.parameters
        self.current_weights = self.params_dict.get("weights",{'Cash':1.0})
        self.index = compute_index_backtrader(self.datas, weights=self.current_weights)
        self.sma_short = bt.indicators.MovingAverageSimple(self.index,
                                                period=self.params_dict.get("short_moving_average",30))
        self.sma_long = bt.indicators.MovingAverageSimple(self.index,
                                                period=self.params_dict.get("long_moving_average",200))


    def rebalance_portfolio(self,positif_sign = 1):
        for asset, target_weight in self.current_weights.items():
            if asset != "Cash":
                if positif_sign == 1:
                    self.order_target_percent(data=asset ,target=target_weight)
                elif positif_sign == -1:
                    self.order_target_percent(data=asset ,target=-target_weight)
                else:
                    self.close(data=asset)

    def notify_order(self, order):

        # if the order fais, we reduce the target weight by 1%
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            asset_name = order.data._name
            if asset_name in self.current_weights:
                self.current_weights[asset_name] = max(0, self.current_weights[asset_name]-0.01)
        
        self.bar_executed = len(self)
        self.order = None 

    def next(self):
        
        if not self.position:

            if self.sma_short[0] > self.sma_long[0]:
                self.rebalance_portfolio(positif_sign=1)

            elif self.sma_short[0] < self.sma_long[0]:
                self.rebalance_portfolio(positif_sign=-1)
            
        else:

            if len(self) >= self.bar_executed + self.params_dict.get("rebalancing_period_momentum",5):

                if self.position.size > 0 and self.sma_short[0] < self.sma_long[0]:
                    self.rebalance_portfolio(positif_sign=0)

                elif self.position.size < 0 and self.sma_short[0] > self.sma_long[0]:
                    self.rebalance_portfolio(positif_sign=0)

    
### Streamlit App

st.title("Backtest Dashboard")

st.divider()

###################### Underlyings / Index

st.subheader("Index")

st.write("Customize your index, the strategies will be implemented using this index")

underlyings_universe_dict =  {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta (Facebook)": "META",
    "NVIDIA": "NVDA",
    "Berkshire Hathaway": "BRK-B",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Walmart": "WMT",
    "Procter & Gamble": "PG",
    "Bank of America": "BAC",
    "Disney": "DIS",
    "Netflix": "NFLX",
    "Adobe": "ADBE",
    "PayPal": "PYPL",
    "Salesforce": "CRM",
    "Intel": "INTC",
    "Coca-Cola": "KO",
    "McDonald's": "MCD",
    "ExxonMobil": "XOM",
    "Pfizer": "PFE",
    "Starbucks": "SBUX",
    "AT&T": "T",
    "Verizon": "VZ",
    "IBM": "IBM",
    "Oracle": "ORCL",
    "Airbus": "AIR.PA",  
    "LVMH": "MC.PA",     
    "L'Oréal": "OR.PA",  
    "TotalEnergies": "TTE.PA",  
    "Sanofi": "SAN.PA",  
    "BNP Paribas": "BNP.PA",  
    "Société Générale": "GLE.PA", 
}

underlyings_selected_name = st.multiselect(label="Underlyings selected",
                                      options=underlyings_universe_dict.keys(),
                                      default=["Microsoft","Tesla","Apple","NVIDIA"])

underlyings_selected = [underlyings_universe_dict[u] for u in underlyings_selected_name]
underlyings_weights = {}
strategies_parameters = {}

with st.expander("Weights",icon="⚖️"):
    
    st.write("*1 - if the sum of the weights is not equal to 1 -> weight = weight / sum(weights)*")
    st.write("*2 - before computing the index, the datas are normalized -> like this, each asset contributes to the index according to its weight, regardless of its absolute price.*")

    for i in range(len(underlyings_selected)):
        
        val = st.number_input(f"Weight of the stock : {underlyings_selected_name[i]}",
                              min_value=0.0,
                              max_value=1.0, 
                              value=round(1/len(underlyings_selected),2)) 
        
        underlyings_weights[underlyings_selected[i]] = val
    
    cash_val = st.number_input(f"Weight of the cash :",
                              min_value=0.0,
                              max_value=1.0, 
                              value=0.0)
    
    underlyings_weights["Cash"] = cash_val 

if len(underlyings_selected) == 0:
    underlyings_weights["Cash"] = 1.0
    
if len(underlyings_selected) > 0 and sum(underlyings_weights.values()) != 1:
    total_weight = sum(underlyings_weights.values())
    for k in underlyings_weights:
        underlyings_weights[k] = underlyings_weights[k] / total_weight

strategies_parameters["weights"] = underlyings_weights

###################### Strategies

st.subheader("Strategies")

strategies_names = ["Buy & Hold","Mean Revertion","Momentum"]
strategies_selected = st.segmented_control("Select the strategies you want to run",
                                            options = strategies_names,
                                            default= strategies_names,
                                            selection_mode="multi",
                                            width="stretch")

with st.expander("Description of the strategies",icon="📖"):
    
    st.write("**Buy & Hold strategy :** Buy the index at the beginning and hold it until the end.")
    st.write("**Mean Revertion strategy :** Buy the index if price <= the average price - 2 standard deviation." \
    " Sell the position if price >= the average price + 5 standard deviation. Close the position if abs(price - average price) <= 0.05 std. " \
    " In this example: the number of std allowed under is 2 and upper is 5 and we close the postion within 0.05 std.")
    st.write("**Momentum strategy :** We compute the short (sma) and long moving averages (lma). " \
    " If we don't have a position : sma > lma -> buy, If sma < lma -> short." \
    " If we have a position, we wait until the next rebalancing, if we are long and sma < lma -> close" \
    " if we are short and sma > lma -> close.")

with st.expander("Parameters of the strategies",icon="⚙️"):
    
    period_mean_reversion = st.number_input("Mean Reversion : period to compute the moving average",
                                        min_value=1,
                                        step=1,
                                        value=20)
                                        
    strategies_parameters["period_mean_reversion"] = period_mean_reversion  

    std_upper_boundary = st.number_input("Mean Reversion : number of std allowed **over** the average price before short selling the index",
                                        min_value=0.0,
                                        step=0.01,
                                        value=5.0)
                                        
    strategies_parameters["std_upper_boundary"] = std_upper_boundary 

    std_lower_boundary = st.number_input("Mean Reversion : number of std allowed **under** the average price before buying the index",
                                        min_value=0.0,
                                        step=0.01,
                                        value=2.0)
                                        
    strategies_parameters["std_lower_boundary"] = std_lower_boundary 

    std_mean_boundary = st.number_input("Mean Reversion : How close to the average in term of std, can we close the position",
                                        min_value=0.0,
                                        step=0.01,
                                        value=0.05)
    strategies_parameters["std_mean_boundary"] = std_mean_boundary 

    short_moving_average = st.number_input("Momentum : Short moving average period",
                                 min_value=1,
                                 step=1,
                                 value=30)
    strategies_parameters["short_moving_average"] = short_moving_average

    long_moving_average = st.number_input("Momentum : Long moving average period",
                                 min_value=1,
                                 step=1,
                                 value=200)
    strategies_parameters["long_moving_average"] = long_moving_average

    rebalancing_period_momentum = st.number_input("Momentum : rebalancing period",
                                 min_value=1,
                                 step=1,
                                 value=5)
    strategies_parameters["rebalancing_period_momentum"] = rebalancing_period_momentum


###################### Time Interval

st.subheader("Time Interval")

left_time, right_time = st.columns(2)

with left_time:
    start_date = st.date_input("Start date",value= dt.datetime(2020,1,1),
                                min_value=dt.datetime(2000,1,1),
                                max_value=dt.datetime.today(),
                                width=300)

with right_time:
    end_date = st.date_input("End date",value= dt.datetime.today(),
                                min_value=dt.datetime(2000,1,1),
                                max_value=dt.datetime.today(),
                                width=300)

if end_date < start_date :
    st.warning("The end date is before the start date")

st.divider()

run_backtest = st.button("**Run Backtest**",width="stretch")

###################### Main
if run_backtest:
    
    if len(underlyings_selected) == 0 or len(strategies_selected) == 0:
        st.info("Please choose at least one underlying and one strategy")
        st.stop()

    if end_date < start_date :
        st.warning("The end date is before the start date")
        st.stop()

    data = get_yahoo_data(underlyings_selected,start_date,end_date)
    data_cerebro = []

    for ticker in underlyings_selected:
        data_per_stock = data.get(ticker)
        data_cerebro.append(bt.feeds.PandasData(dataname=data_per_stock, name=ticker))

    strategies_results = {}
    for strategy in strategies_selected : 

        if strategy == "Buy & Hold":
            strategies_results["Buy & Hold"] = run_strategy(strategy_class=BuyAndHold,
                                                            data=data_cerebro,
                                                            name="Buy & Hold",
                                                            params=strategies_parameters)

        if strategy == "Mean Revertion":
            strategies_results["Mean Revertion"] = run_strategy(strategy_class=MeanRevertion,
                                                            data=data_cerebro,
                                                            name="Mean Revertion",
                                                            params=strategies_parameters)
            
        if strategy == "Momentum":
            strategies_results["Momentum"] = run_strategy(strategy_class=Momentum,
                                                            data=data_cerebro,
                                                            name="Momentum",
                                                            params=strategies_parameters)

###################### Strategies results pyfolio

    strategies_pyfolio = {s : strategies_results[s]["pyfolio"].get_pf_items() for s in strategies_results}
    # strategies_pyfolio["Buy & Hold"][0] => returns
    # strategies_pyfolio["Buy & Hold"][1] => positions
    # strategies_pyfolio["Buy & Hold"][2] => transactions
    # strategies_pyfolio["Buy & Hold"][3] => gross_lev
    
    strategies_returns = {s : strategies_pyfolio[s][0] for s in strategies_results}
    strategies_positions = {s : strategies_pyfolio[s][1] for s in strategies_results}
    strategies_transactions = {s : strategies_pyfolio[s][2] for s in strategies_results}
    strategies_gross_lev = {s : strategies_pyfolio[s][3] for s in strategies_results}  

###################### Metrics
    st.subheader("Metrics")

    metrics = [
        ("Start Value ($)", "start_value", "{:,.2f}"),
        ("Final Value ($)", "final_value", "{:,.2f}"),
        ("Total Return", "total_return_pct", "{:.2f}%"),
        ("Annualized Return", "annualized_return_pct", "{:.2f}%"),
        ("Best Annual Perf","best_year_perf_pct","{:.2f}%"),
        ("Worst Annual Perf","worst_year_perf_pct","{:.2f}%"),
        ("Annualized volatility","annual_std_pct","{:.2f}%"),
        ("Sharpe Ratio", "sharpe", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2f}%"),
        ("Total Trades", "total_trades", "{}"),
        ("Positif Years","number_positive_years","{}"),
        ("Negatif Years","number_negative_years","{}"),
        ("Flat Years","number_flat_years","{}"),
    ]

    table_metrics_list = []
    labels = []
    for label, key, fmt in metrics:
        line = []
        labels.append(label)
        for s in strategies_results:
            val = fmt.format(strategies_results[s][key]) if strategies_results[s][key] is not None else "N/A"
            line.append(val)
        table_metrics_list.append(line)

    table_metric = pd.DataFrame(index=labels,data=table_metrics_list,columns=strategies_selected)
    st.table(table_metric)

###################### Graphics

    st.subheader("Graphics")

    ### Data & Index Performance Graph

    # add each underlying selected to the graph
    data_fig_perf = go.Figure()
    for i,s in enumerate(underlyings_selected):
        data_fig_perf.add_trace(go.Scatter(x=data[s].index,
                                      y=data[s]["Close"],name=underlyings_selected_name[i]))
    
    # # add the customized index to the graph
    customised_index = compute_index(data=data,
                                    weights=underlyings_weights,
                                    underlyings=underlyings_selected)
    
    data_fig_perf.add_trace(go.Scatter(x=customised_index.index,
                                       y=customised_index["index"],
                                       name="Customized Index"))
    

    data_fig_perf.update_layout(
        showlegend = True,
        title = "Indexes performances",
        xaxis_title = "Time",
        yaxis_title="Indexes"
    )
    
    st.plotly_chart(data_fig_perf)

    ### Strategies Performances Graph
    fig_perf = go.Figure()
    for s in strategies_returns:
        s_perf = (1+strategies_returns[s]).cumprod()
        fig_perf.add_trace(go.Scatter(x=strategies_returns[s].index,
                                      y=s_perf,name=s))

    fig_perf.update_layout(
        showlegend = True,
        title = "Strategies performances",
        xaxis_title = "Time",
        yaxis_title="Returns"
    )
    
    st.plotly_chart(fig_perf)

    ### Strategies Histogrammes Graph
    fig_hist = make_subplots(rows=len(strategies_returns)+1,cols=1)
    for i,s in enumerate(strategies_returns):
        fig_hist.append_trace(go.Histogram(x=strategies_returns[s],name=s),i+1,1)

    fig_hist.update_layout(
        showlegend = True,
        title = "Frequencies of strategies returns",
    )
    
    st.plotly_chart(fig_hist)
    

    


















