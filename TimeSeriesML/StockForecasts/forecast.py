from parameters import *
import sys
from stock_prediction import pd
import contextlib

pd.options.mode.chained_assignment = None  # default='warn'
DAYS_BEFORE_LOOKUP = 0
HALF_LOOKUP_STEP = 0
HALF_LOOKUP_STEP_PRICE = 0
QUART_LOOKUP_STEP = 0
QUART_LOOKUP_STEP_PRICE = 0
simdate = "12/31/2070"


# date now
date_now = time.strftime("%Y-%m")
if len(sys.argv) > 2:
    TICKER = sys.argv[1]
    LOOKUP_STEP = int(sys.argv[2])
    HALF_LOOKUP_STEP = int(LOOKUP_STEP/2)
    QUART_LOOKUP_STEP = int(HALF_LOOKUP_STEP/2)
    model_name = f"{date_now}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-activation-{ACTIVATION}"
    if BIDIRECTIONAL:
        model_name += "-b"
if len(sys.argv) > 3:
    simdate = sys.argv[3]
        
import os
from os import path
if not path.exists(os.path.join("results", model_name) + ".h5"):
    print("Model not found: .\\results\\"+model_name)
    quit()

from stock_prediction import create_model, load_data, np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

def predict_prices(model, data, df2, indexVals):
    predicted_prices = []
    for X in range(LOOKUP_STEP*12):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-N_STEPS-X:]
        last_sequence = last_sequence[:N_STEPS]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if SCALE:
            predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]   

        df2.loc[indexVals[-X],'forecast'] = predicted_price
    # add rows for future forecast
    last_date = df2.index[-1]
    new_index = pd.date_range(last_date, periods=LOOKUP_STEP, freq='D')
    df3 = pd.DataFrame(index=new_index, columns=df2.columns)
    df3 = df3.fillna(0)
    return df2.append(df3)

def plot_graph2(future_prices):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.figure(figsize=(18,8))
    plt.title(TICKER+" Stock Price Forecast "+ f"{LOOKUP_STEP}" +" days out", fontsize=16)
    plt.plot(future_prices[f'adjclose'][N_STEPS+1:-LOOKUP_STEP], c='b')
    plt.plot(future_prices[f'forecast'][N_STEPS-LOOKUP_STEP+1:].shift(LOOKUP_STEP), c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    forecast_folder = "forecasts"
    if not os.path.isdir(forecast_folder):
        os.mkdir(forecast_folder)
    filename = os.path.join(forecast_folder, TICKER.lower() + "_" + f"{LOOKUP_STEP}" + "_forecast.png")
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)
    plt.savefig(filename)
    #plt.show()       

# load the data
data = load_data(TICKER, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS, ma_periods=MA_PERIODS, endDate=simdate)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL, activation=ACTIVATION)

model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
#mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
#mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
#print("Mean Absolute Error:", mean_absolute_error)
# predict the future price

# Need to correct scalar for plotting prices without actuals
# get the final dataframe for the testing set
df2 = data['df'].tail(LOOKUP_STEP*12)
df2['forecast'] = 0

indexVals = []
for index in df2.index:
    indexVals.append(index)    
future_prices = predict_prices(model, data, df2, indexVals)

# plot true/pred prices graph
plot_graph2(future_prices)
    
future_price = predict(model, data)

#update prices
future_price = df2['forecast'][df2.index[-1]]
HALF_LOOKUP_STEP_PRICE = df2['forecast'][df2.index[-HALF_LOOKUP_STEP]]
QUART_LOOKUP_STEP_PRICE = df2['forecast'][df2.index[-LOOKUP_STEP+1]]
print(f"Future $ price after {LOOKUP_STEP} days is {future_price:.2f}")
print(f"Future $ price after {HALF_LOOKUP_STEP} days is {HALF_LOOKUP_STEP_PRICE:.2f}")
print(f"Future $ price after 1 day is {QUART_LOOKUP_STEP_PRICE:.2f}")
