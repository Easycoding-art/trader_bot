from Historic_Crypto import HistoricalData
import pandas as pd
import datetime as dt
import os
def difference(start_date, end_date) :
    start_text = start_date.replace("-", " ")
    end_text = end_date.replace("-", " ")
    end_text = end_text.replace(":", " ")
    start = start_text.split(' ')
    end = end_text.split(' ')
    dt1 = dt.datetime(int(start[0]),
                      int(start[1]),
                      int(start[2]),
                      int(start[3]),
                      int(start[4]))
    dt2 = dt.datetime(int(end[0]),
                      int(end[1]),
                      int(end[2]),
                      int(end[3]),
                      int(end[4]), 
                      int(end[5]))
    return (dt2 - dt1).total_seconds()

def get_data(ticker, granularity, start_date) :
    if not os.path.exists(ticker) :
        os.mkdir(ticker)
    new = HistoricalData(ticker, granularity, start_date, verbose = False).retrieve_data()
    new.to_csv(ticker + '/' + ticker + '.csv')
    df = pd.read_csv(ticker + '/' + ticker + '.csv')
    for i in range(len(df)) :
        df.loc[i,'time'] = difference(start_date, df.loc[i,'time'])
    os.remove(ticker + '/' + ticker + '.csv')
    training = df.head(len(new)//10 * 9)
    test = df.tail(len(new)//10)
    training.to_csv(ticker + '/' + ticker + '_training.csv', index= False)
    test.to_csv(ticker + '/' + ticker + '_test.csv', index= False)