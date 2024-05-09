import os
import time
from tensorflow.keras.layers import LSTM


# Window size or the sequence length
N_STEPS = 70
# Lookup step, 1 is the next day
LOOKUP_STEP = 60

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# Shuffle the input data
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.3
# features to use
FEATURE_COLUMNS =  ["adjclose", "sma100", "sma200", "psma100", "psma200", "MACD", "rsi100", "rsi200"]
MA_PERIODS = [50, 100, 200]
# date now
date_now = time.strftime("%Y-%m")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 64
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = True

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
#LOSS = "huber_loss"
#LOSS = "mean_squared_logarithmic_error"
LOSS = "custom_loss"
OPTIMIZER = "adam"
ACTIVATION = "sigmoid"
BATCH_SIZE = 256000
EPOCHS = 400

# Tesla stock market
TICKER = "CVX"
ticker_data_filename = os.path.join("data", f"{TICKER}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-activation-{ACTIVATION}"
if BIDIRECTIONAL:
    model_name += "-b"