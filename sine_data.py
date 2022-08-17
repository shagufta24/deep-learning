import math
import numpy as np
from numpy import arange

lower = -2*math.pi
upper = 2*math.pi
interval = upper - lower
count = 1000
step = interval/count

def min_max_scale(X):
    low = 0
    high = 1
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (high - low) + low
    return X_scaled

def generate_sine_data():
    train_data = []
    for i in arange(lower, upper, step):
        train_data.append([i, math.sin(i)])

    train_data = np.array(train_data)
    
    np.random.shuffle(train_data)
#     train_data = min_max_scale(train_data)

    x_train = train_data[:, :1]
    y_train = train_data[:, 1:2]

    x_train = np.transpose(x_train)
    y_train = np.transpose(y_train)

    x_val = np.random.uniform(low = lower, high = upper, size = 300)
    y_val = np.array([math.sin(x) for x in x_val])

    x_val = x_val.reshape((1, x_val.shape[0]))
    y_val = y_val.reshape((1, y_val.shape[0]))

    return x_train, y_train, x_val, y_val
