#https://medium.com/analytics-vidhya/predicting-stock-price-with-a-feature-fusion-gru-cnn-neural-network-in-pytorch-1220e1231911
import yfinance as yf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from charting import chart_to_image, chart_to_arr
import pandas as pd
import os

#Generating Chart Images - 224 x 224 
#https://github.com/vdyagilev/FALKOR
def test_charting(ticker, dir):
    pho=yf.Ticker(ticker) #"pho.ol"
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
    
    pho_norm = pd.DataFrame(columns=['time','open','high', 'low','close','volume','ewm26','ewm12','macd', 'signal'])
    pho_norm.time = hist.index
    pho_norm.open = norm_open
    pho_norm.close = norm_close
    pho_norm.high = norm_high
    pho_norm.low = norm_low
    pho_norm.volume = hist.Volume.values
    pho_norm.ewm26 = pho_norm.close.ewm(span=26,adjust=False).mean()
    pho_norm.ewm12 = pho_norm.close.ewm(span=12,adjust=False).mean()
    pho_norm.macd = pho_norm.ewm12 - pho_norm.ewm26
    pho_norm.signal = pho_norm.macd.ewm(span=9, adjust=False).mean()
    
    for i in range(0,(len(pho_norm)-30)):
        chart_to_image(pho_norm[i:30+i], dir+'/'+dir+str(i)+'.png')
    
    #chart_to_image(pho_norm.tail(30), 'pho/pho_tail.png')
    #print(pho_norm.shape)

    #arr = chart_to_arr(pho_norm.tail(30))
    #assert arr.shape == (3, 224, 224)
    

def ol_tickers():
    ol = pd.read_csv('ol_ticker.csv', sep='\t', header=None)
    ticker_name = ol[0]
    for i in range(0,len(ticker_name)):
        ticker = ticker_name[i]
        print(ticker)
        ol_ticker = ticker+'.ol'
        df = yf.Ticker(ol_ticker)
        if len(df.history(period="max")) > 30: # only read tickers with more than 30 days history
            if os.path.exists(ticker) and os.path.isdir(ticker):
                os.rmdir(ticker)
            os.mkdir(ticker)
            
            test_charting(ol_ticker, ticker)
        else:
            print("no data for ticker:"+ticker)

if __name__ == '__main__':
    test_charting()