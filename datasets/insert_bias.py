import numpy as np
import pandas as pd

def insert_bias(train_x, train_y):
    df = pd.DataFrame(train_y)
    df['x'] = train_x

    df = df.query('not (Male == 1 and Smiling == 0)')

    train_x = df['x'].to_numpy()
    train_y = df[['Smiling', 'Male']].to_numpy()
    train_y = {'Smiling':train_y[:,0], 'Male':train_y[:,1]}

    return train_x, train_y