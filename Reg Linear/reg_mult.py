import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def h(X, theta):
    return np.dot(X, theta.T)


if __name__ == '__main__':
    data = pd.read_csv('data/trees.csv')
    data.insert(1, 'bias', np.ones(data.shape[0]))
    # X = data.loc[:, ['bias', 'Girth', 'Height']].values
    X = data.loc[:, ['bias', 'Girth']].values
    y = data.loc[:, 'Volume']

    theta     = np.random.rand(X.shape[1])
    theta_aux = np.random.rand(X.shape[1])

    alpha = 0.01
    m     = X.shape[0]
    n     = X.shape[1]

    for _ in range(10000):
        for j in range(n):
            acc = 0
            for i in range(m):
                acc += (h(X[i, :], theta) - y[i]) * X[i, j]
            theta_aux[j] = theta[j] - alpha/m * acc
        theta = theta_aux

    plt.scatter(X[:, 1], y)

    min_x = np.min(X[:, 1])
    max_x = np.max(X[:, 1])

    plt.plot([min_x, max_x], [min_x * theta[1] + theta[0], max_x * theta[1] + theta[0]], c='red')

    print(theta)
