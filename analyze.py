import matplotlib.pyplot as plt
import pandas as pd
def get_graph(pair) :
    df = pd.read_csv(pair+'/'+pair +'_training.csv', index_col=False)
    x = df['time'].to_list()
    y1 = df['low'].to_list()
    y2 = df['high'].to_list()
    plt.plot(x, y1,label = "low")
    plt.plot(x, y2, label = "high")
    plt.legend()
    plt.ylabel(pair.split('-')[1])
    plt.xlabel('seconds')
    plt.title(pair)
    plt.show()

get_graph('BTC-USD')