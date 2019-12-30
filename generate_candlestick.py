#https://medium.com/analytics-vidhya/predicting-stock-price-with-a-feature-fusion-gru-cnn-neural-network-in-pytorch-1220e1231911
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pho=yf.Ticker("pho.ol")
pho.info
hist = pho.history(period="max")

scaler = MinMaxScaler(feature_range = (0, 1))
close = hist.Close.values
close = close.reshape(-1,1)
norm_close = scaler.fit_transform(close)

open = hist.Open.values
open = open.reshape(-1,1)
norm_open = scaler.fit_transform(open)

high = hist.High.values
high = high.reshape(-1,1)
norm_high = scaler.fit_transform(high)

low = hist.Low.values
low = low.reshape(-1,1)
norm_low = scaler.fit_transform(low)

pho_norm = pd.DataFrame(columns=['time','open','high', 'low','close','volume'])
pho_norm.time = hist.index
pho_norm.open = norm_open
pho_norm.close = norm_close
pho_norm.high = norm_high
pho_norm.low = norm_low
pho_norm.volume = hist.Volume.values


#Generating Chart Images - 224 x 224 
#https://github.com/vdyagilev/FALKOR