import tensorflow as tf
import data_miner
import pandas as pd
import numpy as np
import os

def get_train_data(ticker, granularity, start_date) :
    if not os.path.exists(ticker) :
        data_miner.get_data(ticker, granularity, start_date)
    df = pd.read_csv(ticker + '/' + ticker + '_training.csv')
    matrix = df.to_numpy()
    n, m = matrix.shape
    X_pr = matrix[0, 1:7].copy()
    X_pr = np.transpose(X_pr)
    x = np.zeros((50, 1), 'float64')
    y = np.zeros((5, 1), 'float64')
    for i in range(1, n) :
        if i % 10 == 0 :
            a = matrix[i, 1:7].copy()
            a = np.transpose(a)
            y = np.column_stack([y, a])
            X_pr = X_pr.reshape((50, 1))
            x = np.column_stack([x, X_pr])
            X_pr = matrix[i, 1:7].copy()
            X_pr = np.transpose(X_pr)
        else :
            a = matrix[i, 1:7].copy()
            a = np.transpose(a)
            X_pr = np.vstack([X_pr, a])
    x = np.delete(x, 0, axis=1)
    y = np.delete(y, 0, axis=1)
    y = y.T
    x = x.T
    #print(f'x: {x.shape} y: {y.shape}')
    return x, y

def train(ticker,  granularity, start_date) :
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))

    model.compile(optimizer="adam", loss="mean_squared_error")

    X_train, y_train = get_train_data(ticker, granularity, start_date)

    model.fit(X_train, y_train, epochs=1000)

    model.save(ticker + '/' + ticker+'.h5')

#get_train_data('BTC-USD',300,'2023-06-19-00-00')