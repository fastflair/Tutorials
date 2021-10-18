import argparse
import string
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
import pypyodbc
cnxn = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=127.0.0.1;"
                        "Database=db;"
                        "uid=uid;pwd=pwd")

#filename to save plot
filename="e:\\temp\\efficientFrontier.png"

# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument('--mfs', '--names-list', nargs='*', default=['DSMDX', 'RYVIX', 'WFSJX'])
CLI.add_argument('--frets', '--rets-list', nargs='*', default=['0.1', '0.2', '0.15'])

# parse the command line
args = CLI.parse_args()
# access CLI options
print("listMFs: %r" % args.mfs)

endDate="12/31/2070"

TICKERS = []
FORCECAST_RETURNS = []
tickers = args.mfs
FORCECAST_RETURNS = args.frets

allTickers = ""
for TICKER in tickers:
    allTickers += TICKER + " "
    
allReturns = [float(idx) for idx in FORCECAST_RETURNS[0].split(' ')]

#create dictionary
viewdict = {}
count = 0

for tick in allTickers.split():
    viewdict[tick] = float(allReturns[count]);
    count = count + 1

#print(viewdict)
def load_data(tickers):
    out = []
    
    for TICKERL in tickers:
        LTICKER = TICKERL.split()
        for TICKER in LTICKER:
            TICKERS.append(TICKER)
            """
            Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
            Params:
                ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
                n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
                scale (bool): whether to scale prices from 0 to 1, default is True
                shuffle (bool): whether to shuffle the data, default is True
                lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
                test_size (float): ratio for test data, default is 0.2 (20% testing data)
                feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
            """
            # see if ticker is already a loaded stock from yahoo finance
            # load it from yahoo_fin library
            df = si.get_data(TICKER, start_date = "01/01/2019")
            
            # add date as a column
            if "date" not in df.columns:
                df["date"] = df.index
            
            #print(df)
            cols = [(col, TICKER) for col in df.columns]
            df.columns = pd.MultiIndex\
                        .from_tuples(cols,
                                    names=['Attributes', 'Symbols'] )
            out.append(df)
    
    df1 = pd.concat(out, axis=1)
    return df1

# Import data
#print (tickers)
df = load_data(tickers)
df=df.dropna()
#print(TICKERS)

# Closing price
df = df['adjclose']

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import BlackLittermanModel

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
bl = BlackLittermanModel(S, pi="equal", absolute_views=viewdict, omega="default")
rets = bl.bl_returns()

ef = EfficientFrontier(rets, S)
results = ef.max_sharpe()

# generate and draw efficient fronteir
n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

from pypfopt import plotting

# Plot efficient frontier with Monte Carlo sim
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find and plot the tangency portfolio
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Format
ax.set_title("Efficient Frontier Mutual Fund Analysis ("+allTickers.rstrip()+")")
ax.legend()
plt.tight_layout()
#plt.show()
plt.savefig(filename)

for key, value in results.items():
    if value > 0.01:
        print(key, str(round(value, 4)))