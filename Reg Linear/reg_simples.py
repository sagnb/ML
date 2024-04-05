import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/mt_cars.csv')
    x = data.loc[:, 'disp'].values
    y = data.loc[:, 'mpg'].values
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    err_x = x - mean_x
    err_y = y - mean_y
    sum_err = np.sum(err_x * err_y)
    err_x2 = err_x**2
    sum_errx2 = np.sum(err_x2)
    m = sum_err / sum_errx2
    c = mean_y - m*mean_x
    predict = m*x[0]+c
    print(predict)
