#https://medium.com/analytics-vidhya/predicting-stock-price-with-a-feature-fusion-gru-cnn-neural-network-in-pytorch-1220e1231911
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from charting import chart_to_image, chart_to_arr
import pandas as pd

def test_charting():
    pho=yf.Ticker("pho.ol")
    #pho.info
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

    print(pho_norm.shape)
    
    arr = chart_to_arr(pho_norm)
    assert arr.shape == (3, 224, 224)
    chart_to_image(pho_norm, 'test_pho.png')


#Generating Chart Images - 224 x 224 
#https://github.com/vdyagilev/FALKOR

if __name__ == '__main__':
    test_charting()