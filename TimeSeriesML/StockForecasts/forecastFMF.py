import os
from os import path
import sys
import contextlib
import argparse
import string
import numpy as np
from numpy import fft
import pandas as pd
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
import pylab as pl
from datetime import datetime
from pandas_datareader import data as pdr
import pypyodbc
cnxn = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=127.0.0.1;"
                        "Database=db;"
                        "uid=uid;pwd=pwd")
                        
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

now = datetime.now() # current date and time
endDate = now.strftime("%m/%d/%Y")

if len(sys.argv) > 2:
    TICKER = sys.argv[1]
    LOOKUP_STEP = int(sys.argv[2])
if len(sys.argv) > 3:
    endDate = sys.argv[3]
        
def fourierExtrapolation(x, n_predict, period):
    n = x.size
    n_harm = period
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)
    x_notrend = x - p[0] * t
    x_freqdom = fft.fft(x_notrend)
    f = fft.fftfreq(n)
    indexes = list(range(n))
    indexes.sort(key=lambda i: np.absolute(f[i]))

    #indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
    #indexes.reverse()
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n
        phase = np.angle(x_freqdom[i])
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

query = "SELECT [date_time] as date_time, round([price_open]/100.0, 2) as [open], round([price_high]/100.0, 2) as [high],  round([price_low]/100.0, 2) as [low],  round([price_close]/100.0, 2) as [close],  round([adjClose]/100.0, 2) as [adjclose], [volume], '"+TICKER+"' as ticker  FROM [StockTitan].[st].[daily] where stock_id = (select stock_id from [st].[stock] where symbol = '"+TICKER+"') and date_time + 2555 >= '"+endDate+"' and date_time <= '"+endDate+"' and price_close > 0 order by date_time"
df = pd.read_sql_query(query, cnxn)
df = df.set_index('date_time')

NPERIOD = int(len(df.index)/29)

hist = df.loc[:,'close'].values
train = df.loc[:endDate,'close'].values

plt.figure(figsize=(18,8))
n_predict = len(hist) - len(train) + LOOKUP_STEP
extrapolation = fourierExtrapolation(train, n_predict, NPERIOD)
extrapolation[extrapolation < 0] = 0
pl.plot(np.arange(0, hist.size), hist, 'b', label = 'Data', linewidth = 3)
pl.plot(np.arange(0, train.size), train, 'c', label = 'Train', linewidth = 2)
pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'Predict', linewidth = 1)

pl.legend()


forecast_folder = "forecasts"
if not os.path.isdir(forecast_folder):
    os.mkdir(forecast_folder)
filename = os.path.join(forecast_folder, TICKER.lower() + "_" + f"{LOOKUP_STEP}" + "_forecast.png")
with contextlib.suppress(FileNotFoundError):
    os.remove(filename)
plt.savefig(filename)

print(f"Future $ price after {LOOKUP_STEP} days is {extrapolation[-1]:.2f}")
HALF_STEP = int(LOOKUP_STEP/2)
print(f"Future $ price after {HALF_STEP} days is {extrapolation[-HALF_STEP]:.2f}")
print(f"Future $ price after 1 day is {extrapolation[-LOOKUP_STEP]:.2f}")