#Imports, keys here are pandas_datareader allows us to download easily and
#yfinance allows us to get into yahoo
import pandas_datareader.data as data
import yfinance as yf
import pandas as pd
import argparse

import sys, os
from datetime import date
import time

date_now = time.strftime("%Y-%m-%d")
options_data_filename = os.path.join("e:/options/", f"options_screen_{date_now}.csv")

optoinsFile = "e:/options/options_screen.csv"
if os.path.exists(optoinsFile):
  os.remove(optoinsFile)

yf.pdr_override()

calls_or_puts = 'calls'
exp_date_list = ['2021-07-16', '2021-10-15', '2022-01-21', '2023-01-20']
final = pd.DataFrame()

# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument('--stocks', '--names-list', nargs='*', default=['WISH', 'SPCE', 'EVFM'])

# parse the command line
args = CLI.parse_args()

sp_list = []
sp_list = args.stocks

def unusualActivity(calls_or_puts, exp_date, stocklist):
    i = 1
    old_stdout = sys.stdout
    print(exp_date)
    finaldf = pd.DataFrame()
    for TICKER in stocklist:
        for x in TICKER.split(' '):
            if i%250 == 0:
                print(i)
            i +=1
            sys.stdout = open(os.devnull, "w") 
            ticker = yf.Ticker(x)
            try:
                opt = ticker.option_chain(exp_date)
                if calls_or_puts == 'calls':
                    opt.calls.insert(0, 'Exp_Date', exp_date)
                    opt.calls.insert(0, 'Symbol', x)
                    opt.calls.insert(3, 'stock_price', data.get_data_yahoo(x, end_date = date.today())['Close'][-1])
                    opt.calls['V/OI'] = (opt.calls['volume'].astype('float')/opt.calls['openInterest']) 
                    opt.calls['Liquidity'] = (opt.calls['volume'].astype('float')*opt.calls['lastPrice'] + opt.calls['openInterest'].astype('float')*opt.calls['lastPrice']) 
                    opt.calls['ProfitVal'] = (opt.calls['stock_price'] - opt.calls['strike'] - opt.calls['ask']) 
                    opt.calls['PercentIncreaseProfit'] = (-opt.calls['ProfitVal'] / opt.calls['stock_price']*100) 
                    opt.calls['CallPutRatio'] = (opt.calls['volume'].astype('float')+opt.calls['openInterest'])  / (opt.puts['volume'].astype('float')+opt.puts['openInterest'])
                    finaldf = finaldf.append(opt.calls[(opt.calls['Liquidity'] > 10000) & (opt.calls['V/OI'] > 0.5) & (opt.calls['CallPutRatio'] > 2) & (opt.calls['volume'] > 1.5 * opt.calls['openInterest'])])
                    sys.stdout = old_stdout
                elif calls_or_puts == 'puts':
                    opt.puts.insert(0, 'Exp_Date', exp_date)
                    opt.puts.insert(0, 'Symbol', x)
                    opt.puts.insert(3, 'stock_price', data.get_data_yahoo(x, end_date = date.today())['Close'][-1])
                    opt.puts['stock_price'] = data.get_data_yahoo(x, end_date = date.today())['Close'][-1]
                    opt.puts['V/OI'] = (opt.puts['volume'].astype('float')/opt.puts['openInterest'])
                    opt.puts['Liquidity'] = (opt.puts['volume'].astype('float')*opt.puts['lastPrice'] + opt.puts['openInterest'].astype('float')*opt.puts['lastPrice']) 
                    opt.puts['ProfitVal'] = (opt.puts['stock_price'] - opt.puts['strike'] - opt.puts['ask']) 
                    opt.puts['PercentIncreaseProfit'] = (-opt.puts['ProfitVal'] / opt.puts['stock_price']*100) 
                    opt.puts['PutCallRatio'] =  (opt.puts['volume'].astype('float')+opt.puts['openInterest']) / (opt.calls['volume'].astype('float')+opt.calls['openInterest'])
                    finaldf = finaldf.append(opt.puts[(opt.puts['Liquidity'] > 10000) & (opt.puts['V/OI'] > 0.5) & (opt.puts['PutCallRatio'] > 2) & (opt.puts['volume'] > 1.5 * opt.puts['openInterest'])])
                    sys.stdout = old_stdout
                else:
                    print('set calls_or_puts equal to calls or puts retard')
                    break
            except:
                pass
                sys.stdout = old_stdout
    return finaldf

for exp_date in exp_date_list:
    returned = unusualActivity(calls_or_puts, exp_date, sp_list)
    
    returned = returned.drop(columns = ['contractSymbol', 'lastTradeDate', 'contractSize', 'currency'])
    returned.insert(3, 'Distance OTM', returned['stock_price'] - returned['strike'])
    returned['Value'] = (returned['openInterest']+returned['volume'])*returned['bid']*100
    returned = returned.sort_values('V/OI',ascending=False)
    
    final = final.append(returned)

final = final.sort_values('V/OI',ascending=False)
if len(final.index) > 0:
    final.to_csv(options_data_filename)
    final.to_csv(optoinsFile)