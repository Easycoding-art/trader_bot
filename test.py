import tensorflow as tf
import pandas as pd
import numpy as np
import random

def get_test_data(ticker) :
    df = pd.read_csv(ticker + '/' + ticker + '_test.csv')
    matrix = df.to_numpy()
    n, m = matrix.shape
    test_case = random.randint(0, n-10)
    X_pr = matrix[test_case, 1:7].copy()
    X_pr = np.transpose(X_pr)
    x = np.zeros((50, 1), 'float64')
    y = np.zeros((5, 1), 'float64')
    for i in range(test_case+1, test_case+11) :
        if (i-test_case) % 10 == 0 :
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
    return x, y

def test(ticker) :
    model = tf.keras.models.load_model(ticker + '/' + ticker+'.h5')
    X_test, y_test = get_test_data(ticker)
    print(model.predict(X_test))
    print(y_test)